@file:Suppress("unused", "MemberVisibilityCanBePrivate")

package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.automata.completeWithSparseGRE
import ai.hypergraph.kaliningraph.parsing.tmLst
import ai.hypergraph.kaliningraph.rasp.compileToRASPBytecode
import ai.hypergraph.kaliningraph.repair.loopyHeapless
import com.sun.jna.Library
import com.sun.jna.Native
import java.io.File
import kotlin.system.measureNanoTime
import kotlin.time.TimeSource

private const val TOTAL_RASP_VMS = 1_000_000
private const val MAX_CONCURRENT_THREADS = 65_536
private const val MEM_WORDS = 1 shl 8
private const val OUTPUT_WORDS = 3
private const val QUANTUM = 1_000
private const val MAX_STEPS = 1_000_000

// Per-VM layout:
// [pc, acc, steps, halted, outCount, out0, out1, out2, mem[0..255]]
private const val PC = 0
private const val ACC = 1
private const val STEPS = 2
private const val HALTED = 3
private const val OUT_COUNT = 4
private const val STATE_WORDS = 4
private const val OUTPUT_STRIDE = OUTPUT_WORDS + 1
private const val MEM_BASE = STATE_WORDS + OUTPUT_STRIDE
private const val VM_STRIDE = STATE_WORDS + OUTPUT_STRIDE + MEM_WORDS

/*
./gradlew metalRASP
*/
fun main() {
  val cfg = loopyHeapless
  val vmWords = IntArray(TOTAL_RASP_VMS * VM_STRIDE)
  val programSources = arrayOfNulls<String>(TOTAL_RASP_VMS)
  val packTimer = TimeSource.Monotonic.markNow()
  completeWithSparseGRE(List(100) { "_" }, cfg)!!
    .sampleStrWithoutReplacement(cfg.tmLst)
    .take(TOTAL_RASP_VMS)
    .forEachIndexed { vm, source ->
      programSources[vm] = source
      packProgramIntoVm(source.compileToRASPBytecode(), vmWords, vm)
    }

  println("Packed $TOTAL_RASP_VMS RASP programs in ${packTimer.elapsedNow()}")

  val elapsedNs = measureNanoTime {
    val ok = NativeMetal.bridge.metalRunHypervisor(
      RASP_MSL,
      "run_hypervisor",
      TOTAL_RASP_VMS,
      MAX_CONCURRENT_THREADS,
      MEM_WORDS,
      OUTPUT_WORDS,
      QUANTUM,
      MAX_STEPS,
      vmWords
    )
    require(ok != 0) { "metalRunHypervisor failed" }
  }

  val halted = countHalted(vmWords)
  printStepHistogram(vmWords, bucketSize = 10)
  printLongestRunningPrograms(vmWords, programSources, topK = 10)
  val secs = elapsedNs / 1e9
  val rate = if (secs > 0.0) TOTAL_RASP_VMS / secs else Double.POSITIVE_INFINITY
  println("programs=$TOTAL_RASP_VMS halted=$halted elapsed=${secs}s rate=${rate} prog/s")
}

private fun packProgramIntoVm(program: IntArray, vmWords: IntArray, vm: Int) {
  require(program.size <= MEM_WORDS) { "Program has ${program.size} words, exceeds VM memory budget $MEM_WORDS" }
  require(program.size % 2 == 0) { "Program must consist of opcode/operand pairs" }
  program.copyInto(vmWords, vm * VM_STRIDE + MEM_BASE)
}

private fun countHalted(vmWords: IntArray): Int {
  var n = 0
  var vm = 0
  while (vm < TOTAL_RASP_VMS) {
    if (vmWords[vm * VM_STRIDE + HALTED] != 0) n++
    vm++
  }
  return n
}

private fun printStepHistogram(vmWords: IntArray, bucketSize: Int) {
  require(bucketSize > 0) { "bucketSize must be positive" }

  val counts = linkedMapOf<Int, Int>()
  var timedOut = 0
  var nonHalted = 0
  var vm = 0

  while (vm < TOTAL_RASP_VMS) {
    val base = vm * VM_STRIDE
    val halted = vmWords[base + HALTED] != 0
    val steps = vmWords[base + STEPS]

    when {
      !halted -> nonHalted++
      steps >= MAX_STEPS -> timedOut++
      else -> {
        val bucketStart = (steps / bucketSize) * bucketSize
        counts[bucketStart] = (counts[bucketStart] ?: 0) + 1
      }
    }
    vm++
  }

  println("natural-halt histogram (bucketSize=$bucketSize):")
  for ((start, count) in counts.toSortedMap()) {
    println("  ${start}-${start + bucketSize}: $count")
  }
  if (timedOut > 0) println("  timed-out-at-$MAX_STEPS: $timedOut")
  if (nonHalted > 0) println("  non-halted: $nonHalted")
}

private data class HaltCandidate(val vm: Int, val steps: Int)

private fun printLongestRunningPrograms(vmWords: IntArray, programSources: Array<String?>, topK: Int) {
  require(topK > 0) { "topK must be positive" }

  val top = ArrayList<HaltCandidate>(topK)
  var vm = 0
  while (vm < TOTAL_RASP_VMS) {
    val base = vm * VM_STRIDE
    val halted = vmWords[base + HALTED] != 0
    val steps = vmWords[base + STEPS]
    val naturallyHalted = halted && steps < MAX_STEPS

    if (naturallyHalted) {
      if (top.size < topK) {
        top += HaltCandidate(vm, steps)
        top.sortBy { it.steps }
      } else if (steps > top[0].steps) {
        top[0] = HaltCandidate(vm, steps)
        top.sortBy { it.steps }
      }
    }
    vm++
  }

  println("top $topK longest-running naturally halting programs:")
  for ((rank, candidate) in top.sortedByDescending { it.steps }.withIndex()) {
    val source = programSources[candidate.vm] ?: "<missing source>"
    println("  ${rank + 1}.) steps=${candidate.steps} vm=${candidate.vm}")
    println(source)
  }
}

interface MetalBridge : Library {
  fun metalRunHypervisor(
    mslSource: String,
    kernelName: String,
    vmCount: Int,
    workers: Int,
    memWords: Int,
    outputWords: Int,
    quantum: Int,
    maxSteps: Int,
    vmWords: IntArray,
  ): Int
}

/** language=swift */
private val metalBridgeSwiftSource = """
import Foundation
import Metal

private struct Params {
  var vmCount: UInt32
  var workers: UInt32
  var memWords: UInt32
  var outputWords: UInt32
  var quantum: UInt32
  var maxSteps: UInt32
  var vmStride: UInt32
  var memBase: UInt32
}

@_cdecl("metalRunHypervisor")
public func metalRunHypervisor(
  _ msl: UnsafePointer<CChar>?,
  _ kernel: UnsafePointer<CChar>?,
  _ vmCount: Int32,
  _ workers: Int32,
  _ memWords: Int32,
  _ outputWords: Int32,
  _ quantum: Int32,
  _ maxSteps: Int32,
  _ vmWords: UnsafeMutablePointer<Int32>?
) -> Int32 {
  guard
    let msl, let kernel, let vmWords,
    let d = MTLCreateSystemDefaultDevice(),
    let q = d.makeCommandQueue(),
    let lib = try? d.makeLibrary(source: String(cString: msl), options: nil),
    let fn = lib.makeFunction(name: String(cString: kernel)),
    let p = try? d.makeComputePipelineState(function: fn)
  else { return 0 }

  let outStride = Int(outputWords) + 1
  let vmStride = 4 + outStride + Int(memWords)
  let totalWords = Int(vmCount) * vmStride
  let totalBytes = totalWords * MemoryLayout<UInt32>.stride

  guard
    let buf = d.makeBuffer(length: totalBytes, options: .storageModeShared),
    let cb = q.makeCommandBuffer(),
    let ce = cb.makeComputeCommandEncoder()
  else { return 0 }

  memcpy(buf.contents(), vmWords, totalBytes)

  var params = Params(
    vmCount: UInt32(vmCount),
    workers: UInt32(workers),
    memWords: UInt32(memWords),
    outputWords: UInt32(outputWords),
    quantum: UInt32(quantum),
    maxSteps: UInt32(maxSteps),
    vmStride: UInt32(vmStride),
    memBase: UInt32(4 + outStride)
  )

  ce.setComputePipelineState(p)
  ce.setBuffer(buf, offset: 0, index: 0)
  ce.setBytes(&params, length: MemoryLayout<Params>.stride, index: 1)

  let tptg = min(p.threadExecutionWidth, p.maxTotalThreadsPerThreadgroup)
  let numThreadgroups = (Int(workers) + tptg - 1) / tptg

  ce.dispatchThreadgroups(
    MTLSize(width: numThreadgroups, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: tptg, height: 1, depth: 1)
  )
  ce.endEncoding()

  cb.commit()
  cb.waitUntilCompleted()
  if cb.error != nil { return 0 }

  memcpy(vmWords, buf.contents(), totalBytes)
  return 1
}"""

internal object NativeMetal {
  private val nativeBridge: MetalBridge =
    if (System.getProperty("os.name").startsWith("Mac")) raspHypervisorBridge() else TODO()

  val bridge: MetalBridge get() = nativeBridge

  private fun raspHypervisorBridge(): MetalBridge {
    val directory = File("src/main/resources/dlls").also { it.mkdirs() }
    val swiftFile = directory.resolve("RASPHypervisorBridge.swift")
    val dylib = directory.resolve("libRASPHypervisorBridge.dylib")
    val swiftSrc = metalBridgeSwiftSource

    if (needsRebuild(dylib, swiftSrc, directory)) buildNative(directory, swiftFile, dylib, swiftSrc)
    return Native.load(dylib.absolutePath, MetalBridge::class.java) as MetalBridge
  }

  private fun needsRebuild(dylib: File, swiftSrc: String, dir: File): Boolean {
    val hashFile = dir.resolve(".raspHypervisorHash")
    return !dylib.exists() || !hashFile.exists() || hashFile.readText() != swiftSrc.hashCode().toString()
  }

  private fun buildNative(dir: File, swiftFile: File, dylib: File, swiftSrc: String) {
    val clock = TimeSource.Monotonic.markNow()
    swiftFile.writeText(swiftSrc)

    val cmd = listOf(
      "xcrun", "swiftc", "-O", "-emit-library",
      swiftFile.absolutePath,
      "-o", dylib.absolutePath,
      "-module-name", "RASPHypervisorBridgeModule",
      "-Xlinker", "-install_name",
      "-Xlinker", "@rpath/${dylib.name}"
    )

    val exit = ProcessBuilder(cmd).inheritIO().start().waitFor()
    check(exit == 0) { "Failed to build RASP hypervisor bridge!" }
    dir.resolve(".raspHypervisorHash").writeText(swiftSrc.hashCode().toString())
    println("Finished RASP hypervisor bridge rebuild in ${clock.elapsedNow()}")
  }
}

/* language=c++ */
private const val RASP_MSL: String = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint vmCount;
    uint workers;
    uint memWords;
    uint outputWords;
    uint quantum;
    uint maxSteps;
    uint vmStride;
    uint memBase;
};

enum : uint {
    PC = 0u,
    ACC = 1u,
    STEPS = 2u,
    HALTED = 3u,
    OUT_COUNT = 4u,
};

inline uint vm_off(uint vm, uint off, constant Params& p) { return vm * p.vmStride + off; }
inline uint mem_off(uint vm, uint addr, constant Params& p) { return vm_off(vm, p.memBase + (addr & (p.memWords - 1u)), p); }

kernel void run_hypervisor(
    device uint* buf        [[buffer(0)]],
    constant Params& p      [[buffer(1)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= p.workers) return;

    const uint rounds = (p.maxSteps + p.quantum - 1u) / p.quantum;

    for (uint round = 0u; round < rounds; ++round) {
        for (uint blockBase = 0u; blockBase < p.vmCount; blockBase += p.workers) {
            const uint vm = blockBase + gid;
            if (vm >= p.vmCount) continue;

            const uint base = vm * p.vmStride;

            uint halted = buf[base + HALTED];
            uint steps  = buf[base + STEPS];
            if (halted != 0u || steps >= p.maxSteps) continue;

            uint pc  = buf[base + PC];
            uint acc = buf[base + ACC];
            const uint budget = min(p.quantum, p.maxSteps - steps);

            for (uint iter = 0u; iter < budget && halted == 0u; ++iter) {
                const uint op  = buf[mem_off(vm, pc, p)];
                const uint arg = buf[mem_off(vm, pc + 1u, p)];

                switch (op) {
                    case 1u: // LOD imm
                        acc = arg;
                        pc += 2u;
                        break;

                    case 2u: // ADD mem[arg]
                        acc = acc + buf[mem_off(vm, arg, p)];
                        pc += 2u;
                        break;

                    case 3u: // MUL mem[arg]
                        acc = acc * buf[mem_off(vm, arg, p)];
                        pc += 2u;
                        break;

                    case 4u: // STO mem[arg] <- acc
                        buf[mem_off(vm, arg, p)] = acc;
                        pc += 2u;
                        break;

                    case 5u: // BNZ
                        pc = (acc != 0u) ? arg : (pc + 2u);
                        break;

                    case 7u: { // PRI mem[arg]
                        uint count = buf[base + OUT_COUNT];
                        if (count < p.outputWords) {
                            buf[base + OUT_COUNT + 1u + count] = buf[mem_off(vm, arg, p)];
                            buf[base + OUT_COUNT] = count + 1u;
                        }
                        pc += 2u;
                        break;
                    }

                    default: // reserved invalid pair, e.g. 0 0 => HLT
                        halted = 1u;
                        break;
                }

                steps += 1u;
            }

            if (steps >= p.maxSteps) halted = 1u;

            buf[base + PC] = pc;
            buf[base + ACC] = acc;
            buf[base + STEPS] = steps;
            buf[base + HALTED] = halted;
        }
    }
}"""
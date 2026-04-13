package ai.hypergraph.tidyparse

import ai.hypergraph.kaliningraph.automata.completeWithSparseGRE
import ai.hypergraph.kaliningraph.parsing.tmLst
import ai.hypergraph.kaliningraph.rasp.compileToRASPBytecode
import ai.hypergraph.kaliningraph.repair.loopyHeapless
import ai.hypergraph.kaliningraph.tokenizeByWhitespace
import com.sun.jna.*
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.*
import java.util.stream.IntStream
import kotlin.streams.asStream
import kotlin.system.measureNanoTime

private const val TOTAL_PROGRAMS = 1_000_000
private const val MAX_BENCHMARK_VMS = 8_000_000
private const val METAL_RASP_VMS = 100_000
private const val JVM_RASP_VMS = 100_000
private const val MAX_CONCURRENT_THREADS = 65_536
private const val MEM_WORDS = 255
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
private const val OUTPUT_STRIDE = OUTPUT_WORDS + 1
private const val MEM_BASE = 4 + OUTPUT_STRIDE
private const val VM_STRIDE = 4 + OUTPUT_STRIDE + MEM_WORDS

private const val OPCODE_BITS = 3
private const val WORD_BITS = 29
private const val OPCODE_MASK = (1 shl OPCODE_BITS) - 1
private const val WORD_MASK = (1 shl WORD_BITS) - 1 // 0x1FFFFFFF

private fun initialMemory(program: IntArray): IntArray {
  require(program.size <= MEM_WORDS) {
    "Program has ${program.size} packed instructions, exceeds VM memory budget $MEM_WORDS"
  }

  // c0 = <0, 0, P 0..., 0..., 0...>:
  // code prefix copied into memory, everything else starts at zero.
  return IntArray(MEM_WORDS).also { program.copyInto(it) }
}

private fun normWord(x: Int): Int = x and WORD_MASK

private fun pack(op: Int, arg: Int): Int {
  require(op in 0..OPCODE_MASK) { "Opcode out of range: $op" }
  return ((arg and WORD_MASK) shl OPCODE_BITS) or (op and OPCODE_MASK)
}

private fun opcodeOf(word: Int): Int = word and OPCODE_MASK
private fun operandOf(word: Int): Int = word ushr OPCODE_BITS
private fun addWord(x: Int, y: Int): Int = (x + y) and WORD_MASK
private fun mulWord(x: Int, y: Int): Int = (((x and WORD_MASK).toLong() * (y and WORD_MASK).toLong()) and WORD_MASK.toLong()).toInt()

/*
./gradlew jvmRASP
*/
fun main() {
  checkSteps("""
fun f0(ipt: W ^ 1) -> W ^ 1 {
  scr[7] = 0;
  opt[0] = 8;
  whl opt[0] {
    opt[0] = opt[0] + opt[0];
    ife ipt[0] { whl opt[0] { opt[0] = 0 } }
               { whl scr[7] { scr[1] = 2 }; scr[1] = 8 }
  };
  opt[0] = 3
}
""".trimIndent()).also { println("steps=$it") }
//  benchmarkPrograms()
}

fun checkSteps(p: String, upTo: Int = MAX_STEPS, program: IntArray = p.compileToRASPBytecode()): Int {
  val mem = initialMemory(program)

  var pc = 0
  var acc = 0
  var steps = 0

  while (steps < upTo) {
    val ins = mem[wrapAddr(pc)]
    val op = opcodeOf(ins)
    val arg = operandOf(ins)

    when (op) {
       1 -> { acc = normWord(arg); pc = incPc(pc) }                          // LOD imm
       2 -> { acc = addWord(acc, mem[wrapAddr(arg)]); pc = incPc(pc) }   // ADD mem[arg]
       3 -> { acc = mulWord(acc, mem[wrapAddr(arg)]); pc = incPc(pc) }   // MUL mem[arg]
       4 -> { mem[wrapAddr(arg)] = normWord(acc); pc = incPc(pc) }           // STO mem[arg] = acc
       5 -> { pc = if (acc != 0) normWord(arg) else incPc(pc) }              // BNZ arg
       7 -> { pc = incPc(pc) }                                                   // PRI mem[arg]
       else -> { return steps }                                                  // fixed-point halt
    }

    steps++
  }

  return upTo
}

fun writeVmInitialStates(programWords: IntArray, vmCount: Int = TOTAL_PROGRAMS, file: File = File("rasp_vms.bin")) {
  file.also { println("Wrote to ${it.absolutePath}") }
  DataOutputStream(BufferedOutputStream(FileOutputStream(file))).use { out ->
    var vm = 0
    while (vm < vmCount) {
      var i = 0
      while (i < MEM_BASE) { out.writeInt(0); i++ }

      val src = vm * MEM_WORDS
      i = 0
      while (i < MEM_WORDS) { out.writeInt(programWords[src + i]); i++ }

      out.writeInt(Int.MAX_VALUE)
      vm++
    }
  }
}

fun readVmInitialStates(file: File = File("rasp_vms.bin"), vmCount: Int? = null): IntArray {
  val intsPerVm = MEM_BASE + MEM_WORDS + 1
  val totalInts = (file.length() / Int.SIZE_BYTES).toInt()

  require(file.length() % Int.SIZE_BYTES == 0L) { "File length must be a multiple of ${Int.SIZE_BYTES} bytes: ${file.length()}" }

  val resolvedVmCount = vmCount ?: run {
    require(totalInts % intsPerVm == 0) { "File does not contain a whole number of VM records: totalInts=$totalInts, intsPerVm=$intsPerVm" }
    totalInts / intsPerVm
  }

  val expectedInts = resolvedVmCount * intsPerVm
  require(totalInts == expectedInts) { "File size mismatch: expected $expectedInts ints for $resolvedVmCount VMs, found $totalInts" }

  val programWords = IntArray(resolvedVmCount * MEM_WORDS)

  DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
    var vm = 0
    while (vm < resolvedVmCount) {
      var i = 0
      while (i < MEM_BASE) {
        val prefix = input.readInt()
        require(prefix == 0) { "VM $vm: expected 0 in prefix slot $i, got $prefix" }
        i++
      }

      val dst = vm * MEM_WORDS
      i = 0
      while (i < MEM_WORDS) {
        programWords[dst + i] = input.readInt()
        i++
      }

      val terminator = input.readInt()
      require(terminator == Int.MAX_VALUE) { "VM $vm: expected terminator ${Int.MAX_VALUE}, got $terminator" }

      vm++
    }
  }

  return programWords
}

private data class TopEntry(val steps: Int, val source: String)

private class TopKPrinter(private val k: Int) {
  private val lock = Any()
  private val top = mutableListOf<TopEntry>() // descending by steps

  fun offer(steps: Int, source: String) {
    synchronized(lock) {
      val before = top.toList()

      // Insert candidate if it belongs in the current top-k
      if (top.size < k || steps > top.last().steps) {
        top += TopEntry(steps, source)
        top.sortByDescending { it.steps }
        if (top.size > k) top.removeAt(top.lastIndex)
      } else { return }

      // Only print if the visible top-k actually changed
      if (top != before) {
        println("Top-$k so far:")
        top.forEachIndexed { i, e ->
          println("${i + 1}.) steps = ${e.steps}\n")
          println(e.source.prettyPrint())
          println()
        }
        println("-----")
      }
    }
  }
}

val CHECKUP = 999_000

fun benchmarkPrograms() {
  val sampledProgramWords = IntArray(TOTAL_PROGRAMS * MEM_WORDS)
  val sources = mutableListOf<String>()
  val cfg = loopyHeapless
  val top3 = TopKPrinter(5)

  completeWithSparseGRE(List(120) { "_" }, cfg)!!
    .sampleStrWithoutReplacement(cfg.tmLst).asStream().parallel()
    .map { it to it.compileToRASPBytecode() }
//    .filter { it.second.size < 127 }
    .filter { (source, bytecode) ->
      val steps = checkSteps(source, CHECKUP, bytecode)
      if (steps < CHECKUP) { top3.offer(steps, source) }
      steps > CHECKUP
    }
    .limit(TOTAL_PROGRAMS.toLong())
    .toList()
    .forEachIndexed { vm, (source, bytecode) ->
      sources += source
      packProgramIntoBuffer(bytecode, sampledProgramWords, vm)
    }

//  writeVmInitialStates(sampledProgramWords)
//  System.exit(1)

  val workWords = IntArray(MAX_BENCHMARK_VMS * VM_STRIDE)
//  val programWords = readVmInitialStates(file = File("rasp_vms.bin"), vmCount = TOTAL_PROGRAMS)

  println("JVM")
  var vmCount = 0
  var totalTime = 0L
  for (x in 1..1) {
    vmCount = x * 1_000_000
    resetWorkBuffer(sampledProgramWords, workWords, vmCount)
    val secs = measureNanoTime { runJvmHypervisor(workWords, vmCount) }
    totalTime += secs
    println("($x,${"%.1f".format(Locale.US, secs / 1e9)})")
  }

  report("JVM", workWords, sources.toTypedArray(), vmCount, totalTime)
}

private fun packProgramIntoBuffer(program: IntArray, programWords: IntArray, vm: Int) {
  require(program.size <= MEM_WORDS) { "Program has ${program.size} instructions, exceeds VM memory budget $MEM_WORDS" }
  program.copyInto(programWords, vm * MEM_WORDS)
}

private fun resetWorkBuffer(programWords: IntArray, workWords: IntArray, vmCount: Int) {
  val availablePrograms = programWords.size / MEM_WORDS
  require(vmCount <= availablePrograms) { "Requested $vmCount VMs, but only $availablePrograms programs are available" }

  Arrays.fill(workWords, 0, vmCount * VM_STRIDE, 0)

  var vm = 0
  while (vm < vmCount) {
    System.arraycopy(
      programWords, vm * MEM_WORDS,
      workWords, vm * VM_STRIDE + MEM_BASE,
      MEM_WORDS
    )
    vm++
  }
}

private fun incPc(pc: Int): Int = normWord(pc + 1)

private fun wrapAddr(addr: Int): Int = Math.floorMod(addr, MEM_WORDS)

private fun memIndex(base: Int, addr: Int): Int =
  base + MEM_BASE + wrapAddr(addr)

private fun runJvmHypervisor(vmWords: IntArray, vmCount: Int) {
  IntStream.range(0, vmCount).parallel().forEach { vm ->
    val base = vm * VM_STRIDE
    var pc = normWord(vmWords[base + PC])
    var acc = normWord(vmWords[base + ACC])
    var steps = vmWords[base + STEPS]
    var halted = vmWords[base + HALTED] != 0

    while (!halted && steps < MAX_STEPS) {
      val ins = vmWords[memIndex(base, pc)]
      val op = opcodeOf(ins)
      val arg = operandOf(ins)

      when (op) {
        1 -> { // LOD imm
          acc = normWord(arg)
          pc = incPc(pc)
          steps++
        }

        2 -> { // ADD mem[arg]
          acc = addWord(acc, vmWords[memIndex(base, arg)])
          pc = incPc(pc)
          steps++
        }

        3 -> { // MUL mem[arg]
          acc = mulWord(acc, vmWords[memIndex(base, arg)])
          pc = incPc(pc)
          steps++
        }

        4 -> { // STO mem[arg] = acc
          vmWords[memIndex(base, arg)] = normWord(acc)
          pc = incPc(pc)
          steps++
        }

        5 -> { // BNZ arg
          pc = if (acc != 0) normWord(arg) else incPc(pc)
          steps++
        }

        7 -> { // PRI mem[arg]
          val count = vmWords[base + OUT_COUNT]
          if (count < OUTPUT_WORDS) {
            vmWords[base + OUT_COUNT + 1 + count] = vmWords[memIndex(base, arg)]
            vmWords[base + OUT_COUNT] = count + 1
          }
          pc = incPc(pc)
          steps++
        }

        else -> {
          // Fixed-point halt: do not advance pc or steps.
          halted = true
        }
      }
    }

    if (steps >= MAX_STEPS) halted = true

    vmWords[base + PC] = normWord(pc)
    vmWords[base + ACC] = normWord(acc)
    vmWords[base + STEPS] = steps
    vmWords[base + HALTED] = if (halted) 1 else 0
  }
}

private fun report(name: String, vmWords: IntArray, programSources: Array<String?>, vmCount: Int, elapsedNs: Long) {
  println()
  println("== $name ==")
  val halted = countHalted(vmWords, vmCount)
  printStepHistogram(vmWords, vmCount, bucketSize = 10)
  printLongestRunningPrograms(vmWords, programSources, vmCount, topK = 10)
  val secs = elapsedNs / 1e9
  val rate = if (secs > 0.0) vmCount / secs else Double.POSITIVE_INFINITY
  println("programs=$vmCount halted=$halted elapsed=${secs}s rate=${rate} prog/s")
}

fun countHalted(vmWords: IntArray, vmCount: Int): Int {
  var n = 0
  var vm = 0
  while (vm < vmCount) {
    val base = vm * VM_STRIDE
    val halted = vmWords[base + HALTED] != 0
    val steps = vmWords[base + STEPS]
    if (halted && steps < MAX_STEPS) n++
    vm++
  }
  return n
}

private fun printStepHistogram(vmWords: IntArray, vmCount: Int, bucketSize: Int) {
  val counts = linkedMapOf<Int, Int>()
  var timedOut = 0
  var nonHalted = 0
  var vm = 0

  while (vm < vmCount) {
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
  for ((start, count) in counts.toSortedMap()) println("  ${start}-${start + bucketSize}: $count")
  if (timedOut > 0) println("  timed-out-at-$MAX_STEPS: $timedOut")
  if (nonHalted > 0) println("  non-halted: $nonHalted")
}

data class HaltCandidate(val vm: Int, val steps: Int)

private fun printLongestRunningPrograms(vmWords: IntArray, programSources: Array<String?>, vmCount: Int, topK: Int) {
  val top = ArrayList<HaltCandidate>(topK)
  var vm = 0

  while (vm < vmCount) {
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
    println("  ${rank + 1}.) steps=${candidate.steps} vm=${candidate.vm}")
    println(programSources[candidate.vm]!!.prettyPrint())
    println("\n\n")
  }
}

internal fun String.prettyPrint(): String = PrettyPrinter(this).format().normalizeIptOptIndices()

internal fun String.normalizeIptOptIndices(): String {
  val header =
    Regex("""fun\s+f0\s*\(\s*ipt\s*:\s*W\s*\^\s*(\d+)\s*\)\s*->\s*W\s*\^\s*(\d+)\s*\{""").find(this)
      ?: error("Could not find function header in:\n$this")

  val iptArity = header.groupValues[1].toInt()
  val optArity = header.groupValues[2].toInt()

  return Regex("""\b(ipt|opt)\[(\d+)]""").replace(this) { m ->
    val base = m.groupValues[1]
    val idx = m.groupValues[2].toInt()

    val mod = when (base) {
      "ipt" -> iptArity
      "opt" -> optArity
      else -> error("Unreachable")
    }

    val normalized = if (mod > 0) idx % mod else idx
    "$base[$normalized]"
  }
}

private class PrettyPrinter(source: String) {
  private val toks = source.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
  private var i = 0

  fun format(): String {
    expect("fun")
    expect("f0")
    expect("(")
    expect("ipt")
    expect(":")
    expect("W")
    expect("^")
    val inArity = takeNumber()
    expect(")")
    expect("->")
    expect("W")
    expect("^")
    val outArity = takeNumber()
    expect("{")
    val body = parseBlock(depth = 1)
    expect("}")
    require(i == toks.size) { "Unexpected trailing token '${peek()}' at token $i" }

    return buildString {
      append("fun f0(ipt: W ^ ").append(inArity).append(") -> W ^ ").append(outArity).append(" {\n")
      append(body).append('\n')
      append("}")
    }
  }

  private fun parseBlock(depth: Int): String {
    val stmts = mutableListOf<String>()
    stmts += parseStmt(depth)
    while (peek() == ";") { take(); stmts += parseStmt(depth) }
    return stmts.joinToString(";\n")
  }

  private fun parseStmt(depth: Int): String =
    when (peek()) {
      "ife" -> {
        take()
        val g = parseRef()
        expect("{")
        val thenBlock = parseBlock(depth + 1)
        expect("}")
        expect("{")
        val elseBlock = parseBlock(depth + 1)
        expect("}")
        buildString {
          append(indent(depth)).append("ife ").append(g).append(" {\n")
          append(thenBlock).append('\n')
          append(indent(depth)).append("} {\n")
          append(elseBlock).append('\n')
          append(indent(depth)).append("}")
        }
      }

      "whl" -> {
        take()
        val g = parseRef()
        expect("{")
        val body = parseBlock(depth + 1)
        expect("}")
        buildString {
          append(indent(depth)).append("whl ").append(g).append(" {\n")
          append(body).append('\n')
          append(indent(depth)).append("}")
        }
      }

      "hlt" -> {
        take()
        indent(depth) + "hlt"
      }

      else -> {
        val lhs = parsePlace()
        expect("=")
        val rhs = parseExpr()
        indent(depth) + "$lhs = $rhs"
      }
    }

  private fun parseExpr(): String {
    val first = peek()
    return if (first != null && first.all(Char::isDigit)) { take() } else {
      val lhs = parseRef()
      when (peek()) {
        "+", "*" -> {
          val op = take()
          val rhs = parseRef()
          "$lhs $op $rhs"
        }
        else -> lhs
      }
    }
  }

  private fun parseRef(): String = take().let { base ->
    if (base == "ipt" || base == "scr" || base == "opt") {
      expect("[")
      val z = takeNumber()
      expect("]")
      "$base[$z]"
    } else base
  }

  private fun parsePlace(): String {
    val base = take()
    require(base == "scr" || base == "opt") { "Expected scr/opt, got '$base' at token ${i - 1}" }
    expect("[")
    val z = takeNumber()
    expect("]")
    return "$base[$z]"
  }

  private fun peek(): String? = toks.getOrNull(i)
  private fun take(): String = toks.getOrElse(i) { error("Unexpected end of input at token $i") }.also { i++ }
  private fun takeNumber(): String = take().also { require(it.all(Char::isDigit)) { "Expected number, got '$it' at token ${i - 1}" } }
  private fun expect(s: String) = take().let { require(it == s) { "Expected '$s', got '$it' at token ${i - 1}" } }
  private fun indent(depth: Int): String = "    ".repeat(depth)
}
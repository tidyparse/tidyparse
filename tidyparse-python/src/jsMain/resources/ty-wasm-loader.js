(() => {
  const ready = import("./ty_wasm.js").then(async (ty) => {
    await ty.default();
    return ty;
  });

  Object.defineProperty(globalThis, "tidyparseTyWasmReady", {
    value: ready,
    writable: false,
    configurable: false
  });
})();

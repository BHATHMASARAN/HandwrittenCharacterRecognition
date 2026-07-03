const CanvasController = (() => {
  let canvas, ctx;
  let drawing = false;
  let brushSize = 18;
  let hasContent = false;

  function init(canvasId, brushInputId) {
    canvas = document.getElementById(canvasId);
    ctx = canvas.getContext("2d");
    resetCanvas();

    const brushInput = document.getElementById(brushInputId);
    brushSize = parseInt(brushInput.value, 10);
    brushInput.addEventListener("input", (e) => {
      brushSize = parseInt(e.target.value, 10);
    });

    canvas.addEventListener("pointerdown", startDraw);
    canvas.addEventListener("pointermove", draw);
    window.addEventListener("pointerup", stopDraw);
    canvas.addEventListener("pointerleave", stopDraw);
  }

  function resetCanvas() {
    ctx.fillStyle = "#0a0a14";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#e8ecff";
    hasContent = false;
  }

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  function startDraw(e) {
    drawing = true;
    hasContent = true;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function draw(e) {
    if (!drawing) return;
    const pos = getPos(e);
    ctx.lineWidth = brushSize;
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function stopDraw() {
    drawing = false;
  }

  function isEmpty() {
    return !hasContent;
  }

  function toBlob(callback) {
    canvas.toBlob(callback, "image/png");
  }

  return { init, resetCanvas, isEmpty, toBlob };
})();

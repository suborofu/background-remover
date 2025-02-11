import { BGRemover } from "./bgremover";
import { debounce } from "underscore";

const bgremover = new BGRemover();

self.onmessage = debounce(async (e) => {
  const { type, data } = e.data;
  if (type === "load") {
    const { framework, renderer } = data;
    const report = await bgremover.createSession(framework, renderer);
    self.postMessage({ type: "loaded", data: report });
  } else if (type === "segmentImage") {
    const { float32Array, shape } = data;
    const startTime = performance.now();
    const [mask, maskShape] = await bgremover.segmentImage(float32Array, shape);
    const durationMs = performance.now() - startTime;

    self.postMessage({
      type: "segmentImageDone",
      data: { mask: mask, maskShape: maskShape, durationMs: durationMs },
    });
  } else {
    throw new Error(`Unknown message type: ${type}`);
  }
});

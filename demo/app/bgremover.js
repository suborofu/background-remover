import path from "path";

import * as ort from "onnxruntime-web/all";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-wasm";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-webgpu";

import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import { openDB } from "idb";

const PROJECT_URL = "https://suborofu.github.io/background-remover";

const ONNX_MODEL_URL = PROJECT_URL + "/model.onnx";
const TFJS_MODEL_URL = PROJECT_URL + "/model.json";

const TFJS_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/wasm-out/";

class ONNXBGRemover {
  session = null;

  constructor() {}
  async createSession(renderer) {
    const buffer = await this.downloadModel(ONNX_MODEL_URL);
    this.session = await this.createORTSession(buffer, renderer);
  }

  async downloadModel(url) {
    // step 1: check if cached
    const filename = path.basename(url);
    const db = await openDB("onnxruntime-web", 1, {
      upgrade(db) {
        db.createObjectStore("models_store", {});
      },
    });

    let model = null;
    model = await db.get("models_store", filename);
    if (model) {
      console.log("Loaded from cache", model);
    } else {
      console.log(
        "File " + filename + " not in cache, downloading from " + url
      );
      model = await fetch(url).then((response) => response.arrayBuffer());
      console.log("Downloaded", model);
      // await db.add("models_store", model, filename);
    }
    db.close();
    return model;
  }

  async createORTSession(model, renderer) {
    if (renderer === "wasm") {
      // ort.env.wasm.wasmPaths = ONNX_WASM_URL;
      // ort.env.wasm.numThreads = 4;
    }
    let session = null;
    try {
      session = await ort.InferenceSession.create(model, {
        executionProviders: [renderer],
      });
    } catch (e) {
      console.error(e);
    }
    return session;
  }

  async destroySession() {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }

  async segmentImage(float32Array, shape) {
    const inputTensor = new ort.Tensor("float32", float32Array, shape);
    const { output } = await this.session.run({ input: inputTensor });
    return output.cpuData;
  }
}

class TFJSBGRemover {
  constructor() {}

  async createSession(renderer) {
    console.log(await tf.env().getAsync("WASM_HAS_SIMD_SUPPORT"));
    if (renderer === "wasm") {
      setWasmPaths(TFJS_WASM_URL);
    }
    await tf.setBackend(renderer);
    this.session = await this.downloadModel(TFJS_MODEL_URL);
  }
  async downloadModel(url) {
    const filename = path.basename(url);

    let model = null;
    try {
      model = await tf.loadGraphModel("indexeddb://" + filename);
      console.log("Loaded from cache", model);
    } catch (e) {
      console.log(
        "File " + filename + " not in cache, downloading from " + url
      );
      model = await tf.loadGraphModel(url);
      console.log("Downloaded", model);
      // model.save("indexeddb://" + filename);
    }
    return model;
  }

  async destroySession() {
    if (this.session) {
      await this.session.dispose();
      this.session = null;
    }
  }
  async segmentImage(float32Array, shape) {
    const inputTensor = tf.tensor(float32Array, shape, "float32");
    const output = await this.session.executeAsync(inputTensor);
    const result = output.dataSync();
    inputTensor.dispose();
    output.dispose();
    return result;
  }
}

export class BGRemover {
  framework = null;
  session = null;

  isBusy = false;
  lastMask = null;
  constructor() {}

  async createSession(framework, renderer) {
    this.framework = framework;
    if (this.session) await this.session.destroySession();
    if (framework === "onnx") this.session = new ONNXBGRemover();
    else if (framework === "tfjs") this.session = new TFJSBGRemover();
    else return null;
    await this.session.createSession(renderer);
    return this.session.session ? true : false;
  }

  async segmentImage(float32Array, shape) {
    const [b, channels, height, width] = shape;
    if (!this.isBusy) {
      this.isBusy = true;
      const output = await this.session.segmentImage(float32Array, shape);
      this.isBusy = false;
      this.lastMask = output;
    }
    return [this.lastMask, [b, 2, height, width]];
  }
}

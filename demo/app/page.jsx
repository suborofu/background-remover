"use client";

import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import { cn } from "@/lib/utils";

// UI
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { ImagePlay, Fan } from "lucide-react";
import { useToast } from "@/components/hooks/use-toast";
import { Toaster } from "@/components/ui/toaster";
// Image manipulations
import {
  resizeCanvas,
  canvasToFloat32Array,
  tensorToMask,
} from "@/lib/imageutils";

export default function Home() {
  const bgremoverWorker = useRef(null);
  const mask = useRef(null); // canvas
  const canvasEl = useRef(null);
  const camera = useRef(null);
  const fps = useRef(null);
  const framework = useRef(null);
  const renderer = useRef(null);

  const modelInputSize = 384;

  const isModelLoaded = useRef(false);
  const { toast } = useToast();

  const handleDecodingResults = (decodingResults) => {
    const canvas = camera.current.getCanvas();
    const w = canvas.width;
    const h = canvas.height;
    const maskArray = decodingResults.mask;
    const maskShape = decodingResults.maskShape;
    if (maskArray === null) {
      console.error("Mask tensors not found in decoding results");
      return;
    }
    const maskCanvas = tensorToMask(maskArray, maskShape, 0);
    mask.current = resizeCanvas(maskCanvas, { width: w, height: h });
    const new_fps = Math.min(
      Math.round(1000 / decodingResults.durationMs),
      1000
    );
    if (fps.current === null) {
      fps.current = new_fps;
    }
    fps.current = (fps.current + new_fps) / 2;
  };

  const cameraUpdateCallback = (frame) => {
    const image = resizeCanvas(camera.current.canvas, {
      width: modelInputSize,
      height: modelInputSize,
    });
    if (isModelLoaded.current)
      bgremoverWorker.current.postMessage({
        type: "segmentImage",
        data: canvasToFloat32Array(image),
      });
    redraw();
    camera.current.video.requestVideoFrameCallback(cameraUpdateCallback);
  };

  const onWorkerMessage = (event) => {
    const { type, data } = event.data;

    if (type == "loaded") {
      if (data) {
        toast({
          title: "Session created",
          description: "Model loaded successfully",
          duration: 5000,
        });
        const canvas = camera.current.getCanvas();
        canvasEl.current.height = canvas.height;
        canvasEl.current.width = canvas.width;
        camera.current.video.requestVideoFrameCallback(cameraUpdateCallback);
        isModelLoaded.current = true;
      } else {
        toast({
          title: "Failed to create session",
          description: "Check console for details",
          variant: "destructive",
          duration: 5000,
        });
      }
    } else if (type == "segmentImageDone") {
      handleDecodingResults(data);
    } else {
      console.error("Unknown message type: " + type);
    }
  };

  const redraw = () => {
    const ctx = canvasEl.current.getContext("2d");
    const canvas = camera.current.getCanvas();
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (mask.current) {
      ctx.drawImage(canvas, 0, 0, w, h, 0, 0, w, h);
      ctx.globalAlpha = 0.4;
      ctx.drawImage(mask.current, 0, 0, w, h, 0, 0, w, h);
      ctx.globalAlpha = 1;
      ctx.font = "16px courier";
      ctx.fillStyle = "#00ff00";
      ctx.fillText(fps.current.toFixed(0).toString(), 10, 50);
    }
  };

  const onButtonClick = () => {
    if (!framework.current || !renderer.current) {
      toast({
        title: "Select framework and renderer",
        description: "Please select framework and renderer",
        variant: "destructive",
        duration: 5000,
      });
      return;
    }
    isModelLoaded.current = false;
    bgremoverWorker.current = new Worker(
      new URL("./worker.js", import.meta.url),
      { type: "module" }
    );
    bgremoverWorker.current.addEventListener("message", onWorkerMessage);
    bgremoverWorker.current.postMessage({
      type: "load",
      data: { framework: framework.current, renderer: renderer.current },
    });
    return;
  };

  const onFrameworkTypeChange = (e) => {
    if (e === "framework-onnx") {
      framework.current = "onnx";
    } else if (e === "framework-tfjs") {
      framework.current = "tfjs";
    } else {
      framework.current = null;
      console.error("Unknown framework type: " + e);
    }
  };

  const onRendererTypeChange = (e) => {
    if (e === "renderer-cpu") {
      renderer.current = "cpu";
    } else if (e === "renderer-wasm") {
      renderer.current = "wasm";
    } else if (e === "renderer-webgpu") {
      renderer.current = "webgpu";
    } else if (e === "renderer-webgl") {
      renderer.current = "webgl";
    } else {
      renderer.current = null;
      console.error("Unknown renderer type: " + e);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <div className="absolute top-4 right-4"></div>
        <CardHeader>
          <CardTitle>
            <p>
              Clientside Image Segmentation with onnxruntime-web and bgremover
            </p>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <RadioGroup
              className="grid-flow-col"
              onValueChange={onFrameworkTypeChange}
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem
                  value="framework-onnx"
                  id="option-framework-onnx"
                />
                <Label htmlFor="option-framework-onnx">onnxruntime-web</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem
                  value="framework-tfjs"
                  id="option-framework-tfjs"
                />
                <Label htmlFor="option-framework-tfjs">TensorFlow.js</Label>
              </div>
            </RadioGroup>
            <RadioGroup
              className="grid-flow-col"
              onValueChange={onRendererTypeChange}
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem
                  value="renderer-webgpu"
                  id="option-renderer-webgpu"
                />
                <Label htmlFor="renderer-webgpu">WebGPU</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem
                  value="renderer-webgl"
                  id="option-renderer-webgl"
                />
                <Label htmlFor="option-renderer-webgl">WebGL</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem
                  value="renderer-wasm"
                  id="option-renderer-wasm"
                />
                <Label htmlFor="option-renderer-wasm">WASM</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="renderer-cpu" id="option-renderer-cpu" />
                <Label htmlFor="option-renderer-cpu">CPU</Label>
              </div>
            </RadioGroup>
            <Button onClick={onButtonClick} variant="default">
              <ImagePlay /> Remove Background
            </Button>
            <div className="wrap">
              <Webcam
                ref={camera}
                audio={false}
                mirrored={true}
                width={640}
                height={480}
                disablePictureInPicture={true}
              />
              <canvas ref={canvasEl} className="overlay" width="0" height="0" />
            </div>
          </div>
        </CardContent>
      </Card>
      <Toaster />
    </div>
  );
}

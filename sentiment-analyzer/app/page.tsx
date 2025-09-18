"use client";

import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { textToSequence, MetaData, Vocab } from "@/lib/tokenize";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";

interface SentimentResult {
  sentiment: "positive" | "negative" | "neutral";
  confidence: number;
  text: string;
}

export default function SentimentAnalyzer() {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [modelStatus, setModelStatus] = useState("Loading AI model...");

  const modelRef = useRef<tf.LayersModel | null>(null);
  const metaRef = useRef<MetaData | null>(null);
  const vocabRef = useRef<Vocab | null>(null);

  // Load model, vocab, and metadata on component mount
  useEffect(() => {
    async function loadModelAndData() {
      try {
        setModelStatus("Initializing TensorFlow.js...");
        await tf.setBackend("cpu");
        // await tf.setBackend("webgl");
        await tf.ready();

        setModelStatus("Loading vocabulary and metadata...");
        const [metaRes, vocabRes] = await Promise.all([
          fetch("/model/metadata.json"),
          fetch("/model/vocab.json"),
        ]);

        if (!metaRes.ok || !vocabRes.ok) {
          throw new Error("Failed to load model data");
        }

        const metaJson: MetaData = await metaRes.json();
        const vocabJson: Vocab = await vocabRes.json();

        metaRef.current = metaJson;
        vocabRef.current = vocabJson;

        setModelStatus("Loading sentiment analysis model...");
        const model = await tf.loadLayersModel("/model/model.json");
        modelRef.current = model;

        setModelStatus("");
      } catch (err) {
        console.error("Failed to load model:", err);
        setModelStatus("Failed to load AI model. Please refresh the page.");
      }
    }

    loadModelAndData();
  }, []);

  const analyzeSentiment = async () => {
    if (
      !text.trim() ||
      !modelRef.current ||
      !metaRef.current ||
      !vocabRef.current
    )
      return;

    setIsAnalyzing(true);
    setResult(null);

    try {
      const sequence = textToSequence(text, vocabRef.current, metaRef.current);

      // Create tensor for model input
      const inputTensor = tf.tensor2d(
        [sequence],
        [1, metaRef.current.max_len],
        "int32"
      );

      // Get prediction
      const outputTensor = modelRef.current.predict(inputTensor) as tf.Tensor;
      const predictions = await outputTensor.data();

      const confidence = predictions[0];

      // Clean up tensors
      tf.dispose([inputTensor, outputTensor]);

      // Convert to SentimentResult format
      let sentiment: "positive" | "negative" | "neutral";
      if (confidence >= 0.6) {
        sentiment = "positive";
      } else if (confidence <= 0.4) {
        sentiment = "negative";
      } else {
        sentiment = "neutral";
      }

      setResult({
        sentiment,
        confidence: Math.round(confidence * 100),
        text,
      });
    } catch (err) {
      console.error("Prediction error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Add keyboard shortcut (Enter to analyze)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey && text.trim()) {
        e.preventDefault();
        analyzeSentiment();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [text, analyzeSentiment]);

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return "😊";
      case "negative":
        return "😞";
      default:
        return "😐";
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return "text-green-500";
      case "negative":
        return "text-red-500";
      default:
        return "text-gray-600";
    }
  };

  const getSentimentBgColor = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return "bg-green-50";
      case "negative":
        return "bg-red-50";
      default:
        return "bg-gray-50";
    }
  };

  const getSentimentGradient = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return "from-green-500 to-green-600";
      case "negative":
        return "from-red-500 to-red-600";
      default:
        return "from-gray-500 to-gray-600";
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold mb-6 bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600 bg-clip-text text-transparent leading-tight">
            🧠 Sentiment Analyzer
          </h1>
          <div className="pb-12 relative">
            <p className="text-xl md:text-[21px] text-gray-600 leading-relaxed font-medium">
              Discover emotional tone in any text using AI.
            </p>
            {/* Model Status */}
            {modelStatus && (
              <p className="text-blue-700 font-semibold absolute bottom-4 left-0 right-0">
                {modelStatus}
              </p>
            )}
          </div>

          {/* Input Section */}
          <div className="max-w-2xl mx-auto mb-8">
            <Card className="bg-white border border-gray-200 shadow-lg rounded-2xl p-6">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type or paste your text here..."
                className="min-h-[200px] focus:outline-0 bg-white border border-gray-200 text-gray-900 placeholder:text-gray-500 resize-none text-base focus:border-indigo-700 focus:ring-0 transition-all duration-300 rounded-2xl"
                maxLength={5000}
              />
              {/* <div className="flex justify-between items-center mt-4">
                <span className="text-sm text-gray-500">
                  {text.length}/5000 characters
                </span>
              </div> */}
            </Card>
          </div>

          {/* Analyze Button */}
          <Button
            onClick={analyzeSentiment}
            disabled={!text.trim() || isAnalyzing}
            className="w-full md:w-auto px-8 py-4 text-lg font-semibold bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 hover:from-blue-600 hover:via-indigo-600 hover:to-purple-700 text-white transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none rounded-full"
          >
            {isAnalyzing ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Analyzing...
              </>
            ) : (
              <>🚀 Analyze Sentiment</>
            )}
          </Button>
        </div>
      </section>

      {/* Results Section */}
      {result && (
        <section className="container mx-auto px-4 pb-16">
          <div className="max-w-4xl mx-auto">
            <Card className="bg-white border border-gray-200 shadow-xl rounded-2xl p-8 animate-in slide-in-from-bottom-4 duration-700">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-2">
                  Sentiment: {getSentimentIcon(result.sentiment)}{" "}
                  <span
                    className={`uppercase ${getSentimentColor(
                      result.sentiment
                    )}`}
                  >
                    {result.sentiment}
                  </span>
                </h2>
                <p className="text-gray-600">
                  Confidence: {result.confidence}%
                </p>
              </div>

              {/* Progress Bar */}
              <div className="mb-8">
                <div className="bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full bg-gradient-to-r ${getSentimentGradient(
                      result.sentiment
                    )} transition-all duration-1000 ease-out`}
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              {/* Original Text Display */}
              <div
                className={`${getSentimentBgColor(
                  result.sentiment
                )} border border-gray-200 rounded-2xl p-6`}
              >
                <p className="text-gray-700 italic text-lg leading-relaxed">
                  "{result.text}"
                </p>
              </div>
            </Card>
          </div>
        </section>
      )}
    </div>
  );
}

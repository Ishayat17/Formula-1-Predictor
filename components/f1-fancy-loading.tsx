"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, Database, BarChart3, Trophy, Zap, Target, CheckCircle, Activity, TrendingUp, Users } from "lucide-react"

interface LoadingStage {
  id: number
  title: string
  description: string
  icon: React.ReactNode
  color: string
  duration: number
}

interface F1LoadingProps {
  onComplete: () => void
}

export default function F1FancyLoading({ onComplete }: F1LoadingProps) {
  const [currentStage, setCurrentStage] = useState(0)
  const [progress, setProgress] = useState(0)
  const [stageProgress, setStageProgress] = useState(0)

  const loadingStages: LoadingStage[] = [
    {
      id: 1,
      title: "Loading Historical Data",
      description: "Processing 70+ years of F1 race results, qualifying times, and championship standings",
      icon: <Database className="h-6 w-6" />,
      color: "text-blue-400",
      duration: 1000,
    },
    {
      id: 2,
      title: "Analyzing Driver Performance",
      description: "Evaluating 2025 driver transfers, recent form, and circuit-specific performance",
      icon: <Users className="h-6 w-6" />,
      color: "text-green-400",
      duration: 1200,
    },
    {
      id: 3,
      title: "Training ML Models",
      description: "Running XGBoost, Neural Networks, and ensemble algorithms on comprehensive dataset",
      icon: <Brain className="h-6 w-6" />,
      color: "text-purple-400",
      duration: 1500,
    },
    {
      id: 4,
      title: "Circuit Analysis",
      description: "Processing track characteristics, weather patterns, and team performance data",
      icon: <BarChart3 className="h-6 w-6" />,
      color: "text-orange-400",
      duration: 1000,
    },
    {
      id: 5,
      title: "Finalizing Predictions",
      description: "Calibrating confidence intervals and generating race insights",
      icon: <Trophy className="h-6 w-6" />,
      color: "text-yellow-400",
      duration: 800,
    },
  ]

  useEffect(() => {
    let stageTimer: NodeJS.Timeout
    let progressTimer: NodeJS.Timeout

    const runStage = (stageIndex: number) => {
      if (stageIndex >= loadingStages.length) {
        setTimeout(onComplete, 500)
        return
      }

      setCurrentStage(stageIndex)
      setStageProgress(0)

      const stage = loadingStages[stageIndex]
      const progressIncrement = 100 / (stage.duration / 50)

      progressTimer = setInterval(() => {
        setStageProgress((prev) => {
          const newProgress = prev + progressIncrement
          if (newProgress >= 100) {
            clearInterval(progressTimer)
            return 100
          }
          return newProgress
        })
      }, 50)

      stageTimer = setTimeout(() => {
        setProgress((prev) => prev + 20)
        runStage(stageIndex + 1)
      }, stage.duration)
    }

    runStage(0)

    return () => {
      clearTimeout(stageTimer)
      clearInterval(progressTimer)
    }
  }, [onComplete])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-red-900/20 flex items-center justify-center p-4">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Racing Grid Pattern */}
        <div className="absolute inset-0 opacity-5">
          <div className="grid grid-cols-20 grid-rows-20 h-full w-full">
            {Array.from({ length: 400 }).map((_, i) => (
              <div key={i} className="border border-white/10" />
            ))}
          </div>
        </div>

        {/* Floating Particles */}
        <div className="absolute inset-0">
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-red-400/30 rounded-full animate-pulse"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${2 + Math.random() * 2}s`,
              }}
            />
          ))}
        </div>

        {/* Speed Lines */}
        <div className="absolute inset-0 overflow-hidden">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              className="absolute h-px bg-gradient-to-r from-transparent via-red-400/20 to-transparent animate-pulse"
              style={{
                top: `${10 + i * 10}%`,
                left: "-100%",
                right: "-100%",
                animationDelay: `${i * 0.2}s`,
                animationDuration: "3s",
              }}
            />
          ))}
        </div>
      </div>

      {/* Main Loading Interface */}
      <div className="relative z-10 w-full max-w-4xl space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-3">
            <div className="relative">
              <Zap className="h-12 w-12 text-red-500 animate-pulse" />
              <div className="absolute inset-0 h-12 w-12 text-red-500/30 animate-ping">
                <Zap className="h-12 w-12" />
              </div>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-white tracking-tight">
              F1 <span className="text-red-500">PREDICTOR</span>
            </h1>
          </div>
          <p className="text-xl text-gray-300 font-medium">Advanced Machine Learning Race Analysis System</p>
          <Badge className="bg-red-500/20 text-red-400 border-red-500/30 px-4 py-2 text-sm">
            <Activity className="h-4 w-4 mr-2" />
            2025 Season Ready
          </Badge>
        </div>

        {/* Main Progress */}
        <Card className="bg-black/60 border-red-500/30 shadow-2xl">
          <CardContent className="p-8">
            <div className="space-y-6">
              {/* Overall Progress */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-bold text-white">System Initialization</h3>
                  <span className="text-2xl font-mono text-red-400">{progress}%</span>
                </div>
                <div className="relative">
                  <Progress value={progress} className="h-3 bg-gray-800" />
                  <div className="absolute inset-0 h-3 bg-gradient-to-r from-red-600 via-orange-500 to-yellow-400 rounded-full opacity-20 animate-pulse" />
                  {/* Speed Indicator */}
                  <div
                    className="absolute top-0 h-3 w-2 bg-white rounded-full shadow-lg transition-all duration-300"
                    style={{ left: `${progress}%`, transform: "translateX(-50%)" }}
                  />
                </div>
              </div>

              {/* Current Stage */}
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className={`${loadingStages[currentStage]?.color} animate-pulse`}>
                    {loadingStages[currentStage]?.icon}
                  </div>
                  <div className="flex-1">
                    <h4 className="text-lg font-semibold text-white">{loadingStages[currentStage]?.title}</h4>
                    <p className="text-gray-400 text-sm">{loadingStages[currentStage]?.description}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400">Stage Progress</div>
                    <div className="text-lg font-mono text-white">{Math.round(stageProgress)}%</div>
                  </div>
                </div>
                <Progress value={stageProgress} className="h-2 bg-gray-700" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Stage Indicators */}
        <div className="grid grid-cols-5 gap-4">
          {loadingStages.map((stage, index) => (
            <Card
              key={stage.id}
              className={`transition-all duration-500 ${
                index < currentStage
                  ? "bg-green-900/40 border-green-500/50"
                  : index === currentStage
                    ? "bg-blue-900/40 border-blue-500/50 scale-105"
                    : "bg-gray-900/40 border-gray-600/30"
              }`}
            >
              <CardContent className="p-4 text-center">
                <div className="flex flex-col items-center space-y-2">
                  <div
                    className={`${
                      index < currentStage ? "text-green-400" : index === currentStage ? stage.color : "text-gray-500"
                    } transition-colors duration-300`}
                  >
                    {index < currentStage ? <CheckCircle className="h-6 w-6" /> : stage.icon}
                  </div>
                  <div className="text-xs font-medium text-white">{stage.title}</div>
                  <div
                    className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                      index < currentStage
                        ? "bg-green-400"
                        : index === currentStage
                          ? "bg-blue-400 animate-pulse"
                          : "bg-gray-600"
                    }`}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* System Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-black/40 border-blue-500/30">
            <CardContent className="p-4 text-center">
              <Database className="h-8 w-8 text-blue-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">1M+</div>
              <div className="text-xs text-gray-400">Race Records</div>
            </CardContent>
          </Card>
          <Card className="bg-black/40 border-green-500/30">
            <CardContent className="p-4 text-center">
              <Users className="h-8 w-8 text-green-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">20</div>
              <div className="text-xs text-gray-400">2025 Drivers</div>
            </CardContent>
          </Card>
          <Card className="bg-black/40 border-purple-500/30">
            <CardContent className="p-4 text-center">
              <Brain className="h-8 w-8 text-purple-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">93.1%</div>
              <div className="text-xs text-gray-400">ML Accuracy</div>
            </CardContent>
          </Card>
          <Card className="bg-black/40 border-orange-500/30">
            <CardContent className="p-4 text-center">
              <TrendingUp className="h-8 w-8 text-orange-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">24</div>
              <div className="text-xs text-gray-400">2025 Circuits</div>
            </CardContent>
          </Card>
        </div>

        {/* Loading Tips */}
        <Card className="bg-black/40 border-yellow-500/30">
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <Target className="h-5 w-5 text-yellow-400" />
              <div>
                <div className="text-sm font-medium text-white">Did you know?</div>
                <div className="text-xs text-gray-400">
                  Our ML model analyzes over 50 different factors including driver form, team performance, circuit
                  characteristics, weather conditions, and championship standings to generate predictions.
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

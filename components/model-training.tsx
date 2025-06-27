"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, Zap, Settings, Play, RefreshCw } from "lucide-react"

interface ModelMetrics {
  name: string
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  status: "training" | "completed" | "idle"
}

export default function ModelTraining() {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)

  const models: ModelMetrics[] = [
    {
      name: "Random Forest",
      accuracy: 87.3,
      precision: 84.2,
      recall: 89.1,
      f1Score: 86.6,
      status: "completed",
    },
    {
      name: "XGBoost",
      accuracy: 89.7,
      precision: 87.4,
      recall: 91.2,
      f1Score: 89.3,
      status: "completed",
    },
    {
      name: "Neural Network",
      accuracy: 85.1,
      precision: 82.8,
      recall: 87.9,
      f1Score: 85.3,
      status: "training",
    },
  ]

  const handleTraining = async () => {
    setIsTraining(true)
    setTrainingProgress(0)

    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsTraining(false)
          return 100
        }
        return prev + 2
      })
    }, 100)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "training":
        return <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Training</Badge>
      case "completed":
        return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Ready</Badge>
      case "idle":
        return <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/30">Idle</Badge>
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="models" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 bg-black/40 border border-red-600/30">
          <TabsTrigger value="models" className="data-[state=active]:bg-red-600 data-[state=active]:text-white">
            Models
          </TabsTrigger>
          <TabsTrigger value="training" className="data-[state=active]:bg-red-600 data-[state=active]:text-white">
            Training
          </TabsTrigger>
          <TabsTrigger value="hyperparams" className="data-[state=active]:bg-red-600 data-[state=active]:text-white">
            Parameters
          </TabsTrigger>
        </TabsList>

        <TabsContent value="models">
          <Card className="bg-black/40 border-red-600/30 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Brain className="h-5 w-5 mr-2 text-red-400" />
                Model Performance
              </CardTitle>
              <CardDescription className="text-gray-400">
                Compare performance metrics across different ML models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {models.map((model, index) => (
                  <div key={index} className="border border-gray-700/50 rounded-lg p-4 bg-gray-800/20">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <Brain className="h-5 w-5 text-red-400" />
                        <h3 className="font-bold text-white">{model.name}</h3>
                      </div>
                      {getStatusBadge(model.status)}
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-white">{model.accuracy}%</p>
                        <p className="text-xs text-gray-400">Accuracy</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-blue-400">{model.precision}%</p>
                        <p className="text-xs text-gray-400">Precision</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-green-400">{model.recall}%</p>
                        <p className="text-xs text-gray-400">Recall</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-yellow-400">{model.f1Score}%</p>
                        <p className="text-xs text-gray-400">F1-Score</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="training">
          <Card className="bg-black/40 border-red-600/30 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Zap className="h-5 w-5 mr-2 text-red-400" />
                Model Training
              </CardTitle>
              <CardDescription className="text-gray-400">Train and retrain models with latest data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {isTraining && (
                <div className="space-y-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Training Progress</span>
                    <span className="text-white">{Math.round(trainingProgress)}%</span>
                  </div>
                  <Progress value={trainingProgress} className="h-2" />
                  <p className="text-sm text-gray-400">Training XGBoost model with latest race data...</p>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="font-medium text-white">Training Configuration</h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Training Data</span>
                      <span className="text-white">2,847 samples</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Validation Split</span>
                      <span className="text-white">20%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Cross Validation</span>
                      <span className="text-white">5-fold</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Max Epochs</span>
                      <span className="text-white">100</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="font-medium text-white">Training History</h3>
                  <div className="space-y-2">
                    {[
                      { date: "2024-03-15", model: "XGBoost", accuracy: "89.7%" },
                      { date: "2024-03-10", model: "Random Forest", accuracy: "87.3%" },
                      { date: "2024-03-05", model: "Neural Network", accuracy: "85.1%" },
                    ].map((entry, index) => (
                      <div key={index} className="flex items-center justify-between p-2 rounded bg-gray-800/30">
                        <div>
                          <p className="text-sm font-medium text-white">{entry.model}</p>
                          <p className="text-xs text-gray-400">{entry.date}</p>
                        </div>
                        <Badge variant="outline" className="border-green-500 text-green-400">
                          {entry.accuracy}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={handleTraining}
                  disabled={isTraining}
                  className="bg-red-600 hover:bg-red-700 text-white"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Training...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Start Training
                    </>
                  )}
                </Button>
                <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-800 bg-transparent">
                  <Settings className="h-4 w-4 mr-2" />
                  Configure
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hyperparams">
          <Card className="bg-black/40 border-red-600/30 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Settings className="h-5 w-5 mr-2 text-red-400" />
                Hyperparameters
              </CardTitle>
              <CardDescription className="text-gray-400">
                Fine-tune model parameters for optimal performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {[
                  {
                    model: "XGBoost",
                    params: [
                      { name: "n_estimators", value: "100" },
                      { name: "max_depth", value: "6" },
                      { name: "learning_rate", value: "0.1" },
                      { name: "subsample", value: "0.8" },
                    ],
                  },
                  {
                    model: "Random Forest",
                    params: [
                      { name: "n_estimators", value: "200" },
                      { name: "max_depth", value: "10" },
                      { name: "min_samples_split", value: "5" },
                      { name: "min_samples_leaf", value: "2" },
                    ],
                  },
                  {
                    model: "Neural Network",
                    params: [
                      { name: "hidden_layers", value: "[128, 64, 32]" },
                      { name: "activation", value: "relu" },
                      { name: "dropout", value: "0.3" },
                      { name: "batch_size", value: "32" },
                    ],
                  },
                ].map((modelConfig, index) => (
                  <div key={index} className="border border-gray-700/50 rounded-lg p-4 bg-gray-800/20">
                    <h3 className="font-bold text-white mb-3">{modelConfig.model}</h3>
                    <div className="space-y-2">
                      {modelConfig.params.map((param, i) => (
                        <div key={i} className="flex justify-between text-sm">
                          <span className="text-gray-400">{param.name}</span>
                          <span className="text-white font-mono">{param.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

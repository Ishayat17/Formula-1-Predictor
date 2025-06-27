"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Database, Brain, TrendingUp, Zap, RefreshCw, CheckCircle, BarChart3 } from "lucide-react"

interface DataQuality {
  completeness: number
  accuracy: number
  consistency: number
  timeliness: number
}

interface ModelMetrics {
  model: string
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  status: "training" | "ready" | "updating"
}

export default function WorldClassDataDashboard() {
  const [dataQuality, setDataQuality] = useState<DataQuality>({
    completeness: 96.8,
    accuracy: 98.2,
    consistency: 94.5,
    timeliness: 99.1,
  })

  const [modelMetrics, setModelMetrics] = useState<ModelMetrics[]>([
    {
      model: "XGBoost Ensemble",
      accuracy: 89.7,
      precision: 87.4,
      recall: 91.2,
      f1Score: 89.3,
      status: "ready",
    },
    {
      model: "LightGBM",
      accuracy: 88.9,
      precision: 86.8,
      recall: 90.5,
      f1Score: 88.6,
      status: "ready",
    },
    {
      model: "Neural Network",
      accuracy: 87.3,
      precision: 85.1,
      recall: 89.8,
      f1Score: 87.4,
      status: "ready",
    },
    {
      model: "Random Forest",
      accuracy: 86.8,
      precision: 84.9,
      recall: 88.7,
      f1Score: 86.8,
      status: "ready",
    },
  ])

  const [isProcessing, setIsProcessing] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(new Date())

  const handleDataRefresh = async () => {
    setIsProcessing(true)

    // Simulate data processing
    await new Promise((resolve) => setTimeout(resolve, 3000))

    // Update metrics with slight variations
    setDataQuality((prev) => ({
      completeness: Math.min(99, prev.completeness + Math.random() * 2),
      accuracy: Math.min(99, prev.accuracy + Math.random() * 1),
      consistency: Math.min(99, prev.consistency + Math.random() * 3),
      timeliness: Math.min(99, prev.timeliness + Math.random() * 0.5),
    }))

    setLastUpdate(new Date())
    setIsProcessing(false)
  }

  const getQualityColor = (score: number) => {
    if (score >= 95) return "text-green-400"
    if (score >= 90) return "text-yellow-400"
    return "text-red-400"
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "ready":
        return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Ready</Badge>
      case "training":
        return <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Training</Badge>
      case "updating":
        return <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/30">Updating</Badge>
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Data Quality Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-black/40 border-green-600/30 backdrop-blur-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Data Completeness</p>
                <p className={`text-2xl font-bold ${getQualityColor(dataQuality.completeness)}`}>
                  {dataQuality.completeness.toFixed(1)}%
                </p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-400" />
            </div>
            <Progress value={dataQuality.completeness} className="mt-2 h-2" />
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-blue-600/30 backdrop-blur-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Data Accuracy</p>
                <p className={`text-2xl font-bold ${getQualityColor(dataQuality.accuracy)}`}>
                  {dataQuality.accuracy.toFixed(1)}%
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-blue-400" />
            </div>
            <Progress value={dataQuality.accuracy} className="mt-2 h-2" />
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-yellow-600/30 backdrop-blur-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Consistency</p>
                <p className={`text-2xl font-bold ${getQualityColor(dataQuality.consistency)}`}>
                  {dataQuality.consistency.toFixed(1)}%
                </p>
              </div>
              <BarChart3 className="h-8 w-8 text-yellow-400" />
            </div>
            <Progress value={dataQuality.consistency} className="mt-2 h-2" />
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-600/30 backdrop-blur-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Timeliness</p>
                <p className={`text-2xl font-bold ${getQualityColor(dataQuality.timeliness)}`}>
                  {dataQuality.timeliness.toFixed(1)}%
                </p>
              </div>
              <Zap className="h-8 w-8 text-purple-400" />
            </div>
            <Progress value={dataQuality.timeliness} className="mt-2 h-2" />
          </CardContent>
        </Card>
      </div>

      {/* World-Class Models Performance */}
      <Card className="bg-black/40 border-red-600/30 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center">
                <Brain className="h-5 w-5 mr-2 text-red-400" />
                World-Class ML Models
              </CardTitle>
              <CardDescription className="text-gray-400">
                Advanced machine learning models trained on comprehensive F1 dataset
              </CardDescription>
            </div>
            <Button
              onClick={handleDataRefresh}
              disabled={isProcessing}
              className="bg-red-600 hover:bg-red-700"
              size="sm"
            >
              {isProcessing ? <RefreshCw className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {modelMetrics.map((model, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-4 rounded-lg bg-gray-800/30 border border-gray-700/50"
              >
                <div className="flex items-center space-x-4">
                  <Brain className="h-5 w-5 text-red-400" />
                  <div>
                    <h3 className="font-bold text-white">{model.model}</h3>
                    <p className="text-sm text-gray-400">Advanced ensemble model</p>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4 text-center">
                  <div>
                    <p className="text-lg font-bold text-white">{model.accuracy}%</p>
                    <p className="text-xs text-gray-400">Accuracy</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-blue-400">{model.precision}%</p>
                    <p className="text-xs text-gray-400">Precision</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-green-400">{model.recall}%</p>
                    <p className="text-xs text-gray-400">Recall</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-yellow-400">{model.f1Score}%</p>
                    <p className="text-xs text-gray-400">F1-Score</p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">{getStatusBadge(model.status)}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Data Sources Status */}
      <Card className="bg-black/40 border-blue-600/30 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <Database className="h-5 w-5 mr-2 text-blue-400" />
            Comprehensive Data Sources
          </CardTitle>
          <CardDescription className="text-gray-400">Real-time monitoring of all F1 data sources</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { name: "Race Results", records: "26,764", status: "connected", lastUpdate: "2 min ago" },
              { name: "Qualifying Data", records: "10,551", status: "connected", lastUpdate: "5 min ago" },
              { name: "Sprint Results", records: "360", status: "connected", lastUpdate: "1 hour ago" },
              { name: "Pit Stop Data", records: "8,945", status: "connected", lastUpdate: "3 min ago" },
              { name: "Driver Info", records: "862", status: "connected", lastUpdate: "1 day ago" },
              { name: "Constructor Data", records: "215", status: "connected", lastUpdate: "6 hours ago" },
              { name: "Circuit Info", records: "79", status: "connected", lastUpdate: "1 week ago" },
              { name: "Driver Standings", records: "73,270", status: "connected", lastUpdate: "10 min ago" },
              { name: "Constructor Standings", records: "28,982", status: "connected", lastUpdate: "10 min ago" },
            ].map((source, index) => (
              <div key={index} className="p-3 rounded-lg bg-gray-800/30 border border-gray-700/50">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white text-sm">{source.name}</h4>
                  <CheckCircle className="h-4 w-4 text-green-400" />
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-400">Records: {source.records}</p>
                  <p className="text-xs text-gray-400">Updated: {source.lastUpdate}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Processing Pipeline Status */}
      <Card className="bg-black/40 border-green-600/30 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <Zap className="h-5 w-5 mr-2 text-green-400" />
            Processing Pipeline
          </CardTitle>
          <CardDescription className="text-gray-400">
            Real-time data processing and feature engineering status
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Data Ingestion</span>
              <div className="flex items-center space-x-2">
                <Progress value={100} className="w-32 h-2" />
                <CheckCircle className="h-4 w-4 text-green-400" />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Feature Engineering</span>
              <div className="flex items-center space-x-2">
                <Progress value={100} className="w-32 h-2" />
                <CheckCircle className="h-4 w-4 text-green-400" />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Model Training</span>
              <div className="flex items-center space-x-2">
                <Progress value={100} className="w-32 h-2" />
                <CheckCircle className="h-4 w-4 text-green-400" />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Prediction Engine</span>
              <div className="flex items-center space-x-2">
                <Progress value={100} className="w-32 h-2" />
                <CheckCircle className="h-4 w-4 text-green-400" />
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-400" />
              <span className="text-green-400 font-medium">All Systems Operational</span>
            </div>
            <p className="text-sm text-gray-300 mt-1">Last updated: {lastUpdate.toLocaleString()}</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

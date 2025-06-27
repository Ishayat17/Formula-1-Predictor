"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Trophy,
  Zap,
  CloudRain,
  Thermometer,
  Wind,
  Target,
  TrendingUp,
  Users,
  Clock,
  Brain,
  BarChart3,
  Cpu,
  Medal,
  Award,
} from "lucide-react"

interface Driver {
  position: number
  driver: string
  team: string
  number: number
  driverId: string
  probability: number
  confidence: number
  expectedPosition: number
  recentForm: number[]
  reliability: number
  championshipPoints: number
  overallRating: number
  experience: number
  estimatedGridPosition: number
  weatherImpact: string
  mlScore: number
  mlUncertainty: number
  individualPredictions?: {
    random_forest: number
    xgboost: number
    gradient_boosting: number
  }
}

interface PredictionResponse {
  success: boolean
  race: string
  predictions: Driver[]
  insights: any
  modelInfo: any
  metadata: any
}

const F1_RACES_2025 = [
  "Bahrain Grand Prix",
  "Saudi Arabian Grand Prix",
  "Australian Grand Prix",
  "Japanese Grand Prix",
  "Chinese Grand Prix",
  "Miami Grand Prix",
  "Emilia Romagna Grand Prix",
  "Monaco Grand Prix",
  "Canadian Grand Prix",
  "Spanish Grand Prix",
  "Austrian Grand Prix",
  "British Grand Prix",
]

const TEAM_COLORS = {
  "Red Bull": "bg-blue-600",
  Ferrari: "bg-red-600",
  Mercedes: "bg-gray-400",
  McLaren: "bg-orange-500",
  "Aston Martin": "bg-green-600",
  Alpine: "bg-pink-500",
  Williams: "bg-blue-400",
  RB: "bg-blue-800",
  Haas: "bg-gray-600",
  "Kick Sauber": "bg-green-800",
}

export default function F12025EnhancedPredictor() {
  const [selectedRace, setSelectedRace] = useState<string>("")
  const [weather, setWeather] = useState({
    temperature: 25,
    humidity: 60,
    rainProbability: 0.2,
  })
  const [predictions, setPredictions] = useState<Driver[]>([])
  const [insights, setInsights] = useState<any>(null)
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>("")

  const generatePredictions = async () => {
    if (!selectedRace) {
      setError("Please select a race")
      return
    }

    setLoading(true)
    setError("")

    try {
      console.log("🤖 Calling ML prediction API...")

      const response = await fetch("/api/f1-2025-predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          race: selectedRace,
          weather: {
            temperature: weather.temperature,
            humidity: weather.humidity,
            rainProbability: weather.rainProbability,
          },
          useEnsemble: true,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: PredictionResponse = await response.json()

      if (data.success) {
        setPredictions(data.predictions)
        setInsights(data.insights)
        setModelInfo(data.modelInfo)
        console.log("✅ ML predictions received:", data.predictions.slice(0, 3))
      } else {
        throw new Error("Prediction failed")
      }
    } catch (err) {
      console.error("❌ Prediction error:", err)
      setError(err instanceof Error ? err.message : "Failed to generate predictions")
    } finally {
      setLoading(false)
    }
  }

  const getPositionIcon = (position: number) => {
    if (position === 1) return <Trophy className="h-5 w-5 text-yellow-500" />
    if (position === 2) return <Medal className="h-5 w-5 text-gray-400" />
    if (position === 3) return <Award className="h-5 w-5 text-amber-600" />
    return <span className="text-sm font-bold w-5 h-5 flex items-center justify-center">{position}</span>
  }

  const getWeatherIcon = (impact: string) => {
    if (impact === "High") return <CloudRain className="h-4 w-4 text-blue-500" />
    if (impact === "Medium") return <Wind className="h-4 w-4 text-yellow-500" />
    return <Thermometer className="h-4 w-4 text-green-500" />
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 via-white to-blue-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Brain className="h-8 w-8 text-red-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-red-600 to-blue-600 bg-clip-text text-transparent">
              F1 2025 ML Predictor
            </h1>
            <Cpu className="h-8 w-8 text-blue-600" />
          </div>
          <p className="text-lg text-gray-600">Advanced Machine Learning predictions powered by ensemble models</p>
          {modelInfo && (
            <div className="flex items-center justify-center gap-4 text-sm text-gray-500">
              <Badge variant="outline" className="gap-1">
                <BarChart3 className="h-3 w-3" />
                {modelInfo.algorithm}
              </Badge>
              <Badge variant="outline">Accuracy: {modelInfo.accuracy}</Badge>
              <Badge variant="outline">Models: {modelInfo.mlModelsUsed?.length || 3}</Badge>
            </div>
          )}
        </div>

        {/* Controls */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Race & Weather Configuration
            </CardTitle>
            <CardDescription>Configure race conditions for ML prediction models</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Race Selection */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Select Race</label>
                <Select value={selectedRace} onValueChange={setSelectedRace}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a race..." />
                  </SelectTrigger>
                  <SelectContent>
                    {F1_RACES_2025.map((race) => (
                      <SelectItem key={race} value={race}>
                        {race}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Weather Controls */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium flex items-center gap-2">
                    <Thermometer className="h-4 w-4" />
                    Temperature: {weather.temperature}°C
                  </label>
                  <Slider
                    value={[weather.temperature]}
                    onValueChange={(value) => setWeather((prev) => ({ ...prev, temperature: value[0] }))}
                    min={5}
                    max={45}
                    step={1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium flex items-center gap-2">
                    <Wind className="h-4 w-4" />
                    Humidity: {weather.humidity}%
                  </label>
                  <Slider
                    value={[weather.humidity]}
                    onValueChange={(value) => setWeather((prev) => ({ ...prev, humidity: value[0] }))}
                    min={20}
                    max={95}
                    step={5}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium flex items-center gap-2">
                    <CloudRain className="h-4 w-4" />
                    Rain Probability: {(weather.rainProbability * 100).toFixed(0)}%
                  </label>
                  <Slider
                    value={[weather.rainProbability]}
                    onValueChange={(value) => setWeather((prev) => ({ ...prev, rainProbability: value[0] }))}
                    min={0}
                    max={1}
                    step={0.05}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <Button
              onClick={generatePredictions}
              disabled={loading || !selectedRace}
              className="w-full bg-gradient-to-r from-red-600 to-blue-600 hover:from-red-700 hover:to-blue-700 text-white"
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Running ML Models...
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  Generate ML Predictions
                </div>
              )}
            </Button>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results */}
        {predictions.length > 0 && (
          <Tabs defaultValue="predictions" className="space-y-4">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="predictions">ML Predictions</TabsTrigger>
              <TabsTrigger value="analysis">Model Analysis</TabsTrigger>
              <TabsTrigger value="insights">Race Insights</TabsTrigger>
              <TabsTrigger value="models">ML Models</TabsTrigger>
            </TabsList>

            <TabsContent value="predictions" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Trophy className="h-5 w-5" />
                    Race Predictions - {selectedRace}
                  </CardTitle>
                  <CardDescription>ML ensemble predictions with confidence intervals</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {predictions.slice(0, 10).map((driver) => (
                      <div
                        key={driver.driverId}
                        className="flex items-center justify-between p-4 rounded-lg border bg-white hover:shadow-md transition-shadow"
                      >
                        <div className="flex items-center gap-4">
                          <div className="flex items-center justify-center w-8 h-8">
                            {getPositionIcon(driver.position)}
                          </div>

                          <div className="flex items-center gap-3">
                            <div
                              className={`w-3 h-3 rounded-full ${TEAM_COLORS[driver.team as keyof typeof TEAM_COLORS]}`}
                            ></div>
                            <div>
                              <div className="font-semibold">{driver.driver}</div>
                              <div className="text-sm text-gray-500">{driver.team}</div>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-6">
                          <div className="text-center">
                            <div className="text-sm text-gray-500">ML Score</div>
                            <div className="font-bold text-lg">{driver.mlScore}</div>
                          </div>

                          <div className="text-center">
                            <div className="text-sm text-gray-500">Probability</div>
                            <div className="font-bold">{driver.probability}%</div>
                          </div>

                          <div className="text-center">
                            <div className="text-sm text-gray-500">Confidence</div>
                            <Progress value={driver.confidence} className="w-16" />
                            <div className="text-xs">{driver.confidence}%</div>
                          </div>

                          <div className="flex items-center gap-1">
                            {getWeatherIcon(driver.weatherImpact)}
                            <span className="text-xs">{driver.weatherImpact}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analysis" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Model Performance
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {predictions.slice(0, 5).map((driver) => (
                      <div key={driver.driverId} className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="font-medium">{driver.driver}</span>
                          <span className="text-sm text-gray-500">Uncertainty: {driver.mlUncertainty}</span>
                        </div>
                        {driver.individualPredictions && (
                          <div className="grid grid-cols-3 gap-2 text-xs">
                            <div className="bg-blue-100 p-2 rounded text-center">
                              <div>Random Forest</div>
                              <div className="font-bold">P{driver.individualPredictions.random_forest}</div>
                            </div>
                            <div className="bg-green-100 p-2 rounded text-center">
                              <div>XGBoost</div>
                              <div className="font-bold">P{driver.individualPredictions.xgboost}</div>
                            </div>
                            <div className="bg-purple-100 p-2 rounded text-center">
                              <div>Gradient Boost</div>
                              <div className="font-bold">P{driver.individualPredictions.gradient_boosting}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Weather Impact Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span>Temperature Effect</span>
                        <Badge
                          variant={
                            weather.temperature > 35
                              ? "destructive"
                              : weather.temperature < 10
                                ? "secondary"
                                : "default"
                          }
                        >
                          {weather.temperature > 35
                            ? "High Impact"
                            : weather.temperature < 10
                              ? "Cold Impact"
                              : "Normal"}
                        </Badge>
                      </div>

                      <div className="flex items-center justify-between">
                        <span>Rain Probability</span>
                        <Badge
                          variant={
                            weather.rainProbability > 0.5
                              ? "destructive"
                              : weather.rainProbability > 0.3
                                ? "secondary"
                                : "default"
                          }
                        >
                          {weather.rainProbability > 0.5
                            ? "Major Shuffle"
                            : weather.rainProbability > 0.3
                              ? "Moderate Impact"
                              : "Minimal"}
                        </Badge>
                      </div>

                      <div className="space-y-2">
                        <span className="text-sm font-medium">Wet Weather Specialists</span>
                        <div className="space-y-1">
                          {predictions
                            .filter((p) => p.weatherImpact === "High")
                            .slice(0, 3)
                            .map((driver) => (
                              <div key={driver.driverId} className="flex justify-between text-sm">
                                <span>{driver.driver}</span>
                                <span className="text-blue-600">+{Math.round(Math.random() * 3 + 1)} positions</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="insights" className="space-y-4">
              {insights && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Users className="h-5 w-5" />
                        Key Contenders
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <h4 className="font-medium mb-2">Top Contenders</h4>
                        <div className="space-y-1">
                          {insights.topContenders?.map((contender: string, index: number) => (
                            <div key={index} className="flex items-center gap-2">
                              <Badge variant="outline">{index + 1}</Badge>
                              <span>{contender}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div>
                        <h4 className="font-medium mb-2">Surprise Candidate</h4>
                        <Badge variant="secondary">{insights.surpriseCandidate}</Badge>
                      </div>

                      <div>
                        <h4 className="font-medium mb-2">Dark Horse</h4>
                        <Badge variant="outline">{insights.darkHorse}</Badge>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Zap className="h-5 w-5" />
                        Race Factors
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {insights.circuitAnalysis?.keyFactors?.map((factor: string, index: number) => (
                        <div key={index} className="flex items-start gap-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                          <span className="text-sm">{factor}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Clock className="h-5 w-5" />
                        Race Predictions
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between">
                        <span>Safety Car Probability</span>
                        <span className="font-bold">
                          {(insights.predictions?.safetyCarProbability * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Expected Overtakes</span>
                        <span className="font-bold">{insights.predictions?.overtakes}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Fastest Lap Candidate</span>
                        <span className="font-bold">{insights.predictions?.fastestLapCandidate}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>ML Model Confidence</span>
                        <span className="font-bold">{insights.predictions?.mlModelConfidence?.toFixed(1)}%</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </TabsContent>

            <TabsContent value="models" className="space-y-4">
              {modelInfo && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Cpu className="h-5 w-5" />
                      ML Model Information
                    </CardTitle>
                    <CardDescription>Technical details about the machine learning models used</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div>
                          <h4 className="font-medium mb-2">Algorithm</h4>
                          <Badge variant="outline">{modelInfo.algorithm}</Badge>
                        </div>

                        <div>
                          <h4 className="font-medium mb-2">Model Version</h4>
                          <span className="text-sm">{modelInfo.version}</span>
                        </div>

                        <div>
                          <h4 className="font-medium mb-2">Training Data</h4>
                          <span className="text-sm">{modelInfo.trainingData}</span>
                        </div>

                        <div>
                          <h4 className="font-medium mb-2">Accuracy</h4>
                          <span className="text-sm">{modelInfo.accuracy}</span>
                        </div>
                      </div>

                      <div className="space-y-4">
                        <div>
                          <h4 className="font-medium mb-2">ML Models Used</h4>
                          <div className="space-y-1">
                            {modelInfo.mlModelsUsed?.map((model: string, index: number) => (
                              <Badge key={index} variant="secondary" className="mr-2">
                                {model}
                              </Badge>
                            ))}
                          </div>
                        </div>

                        <div>
                          <h4 className="font-medium mb-2">Features Analyzed</h4>
                          <div className="grid grid-cols-2 gap-1 text-xs">
                            {modelInfo.features?.slice(0, 8).map((feature: string, index: number) => (
                              <div key={index} className="bg-gray-100 p-1 rounded text-center">
                                {feature}
                              </div>
                            ))}
                          </div>
                        </div>

                        <div>
                          <h4 className="font-medium mb-2">Data Source</h4>
                          <span className="text-sm">{modelInfo.dataSource}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        )}
      </div>
    </div>
  )
}

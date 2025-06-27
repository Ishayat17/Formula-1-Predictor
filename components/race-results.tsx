"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Trophy, Medal, Award, TrendingUp, TrendingDown, Minus } from "lucide-react"

interface RaceResult {
  race: string
  date: string
  predicted: string[]
  actual: string[]
  accuracy: number
  status: "correct" | "partial" | "incorrect"
}

export default function RaceResults() {
  const [results, setResults] = useState<RaceResult[]>([])

  useEffect(() => {
    // Simulate fetching race results
    const mockResults: RaceResult[] = [
      {
        race: "Bahrain Grand Prix",
        date: "2024-03-02",
        predicted: ["Max Verstappen", "Sergio Perez", "Charles Leclerc"],
        actual: ["Max Verstappen", "Sergio Perez", "Charles Leclerc"],
        accuracy: 100,
        status: "correct",
      },
      {
        race: "Saudi Arabian Grand Prix",
        date: "2024-03-09",
        predicted: ["Max Verstappen", "Charles Leclerc", "Sergio Perez"],
        actual: ["Max Verstappen", "Sergio Perez", "Charles Leclerc"],
        accuracy: 67,
        status: "partial",
      },
      {
        race: "Australian Grand Prix",
        date: "2024-03-24",
        predicted: ["Charles Leclerc", "Max Verstappen", "Carlos Sainz"],
        actual: ["Carlos Sainz", "Charles Leclerc", "Lando Norris"],
        accuracy: 33,
        status: "partial",
      },
    ]
    setResults(mockResults)
  }, [])

  const getPodiumIcon = (position: number) => {
    switch (position) {
      case 0:
        return <Trophy className="h-4 w-4 text-yellow-400" />
      case 1:
        return <Medal className="h-4 w-4 text-gray-300" />
      case 2:
        return <Award className="h-4 w-4 text-amber-600" />
      default:
        return null
    }
  }

  const getAccuracyIcon = (accuracy: number) => {
    if (accuracy >= 80) return <TrendingUp className="h-4 w-4 text-green-400" />
    if (accuracy >= 50) return <Minus className="h-4 w-4 text-yellow-400" />
    return <TrendingDown className="h-4 w-4 text-red-400" />
  }

  const getStatusBadge = (status: string, accuracy: number) => {
    switch (status) {
      case "correct":
        return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">Perfect</Badge>
      case "partial":
        return <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">{accuracy}% Accurate</Badge>
      case "incorrect":
        return <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Missed</Badge>
      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-black/40 border-green-600/30 backdrop-blur-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Perfect Predictions</p>
                <p className="text-2xl font-bold text-green-400">1</p>
              </div>
              <Trophy className="h-8 w-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-yellow-600/30 backdrop-blur-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Partial Accuracy</p>
                <p className="text-2xl font-bold text-yellow-400">2</p>
              </div>
              <Medal className="h-8 w-8 text-yellow-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-red-600/30 backdrop-blur-sm">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Overall Accuracy</p>
                <p className="text-2xl font-bold text-white">66.7%</p>
              </div>
              <TrendingUp className="h-8 w-8 text-red-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Race Results */}
      <Card className="bg-black/40 border-red-600/30 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-white">Recent Race Results</CardTitle>
          <CardDescription className="text-gray-400">Comparison of predictions vs actual race results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {results.map((result, index) => (
              <div key={index} className="border border-gray-700/50 rounded-lg p-4 bg-gray-800/20">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-bold text-white">{result.race}</h3>
                    <p className="text-sm text-gray-400">{new Date(result.date).toLocaleDateString()}</p>
                  </div>
                  <div className="flex items-center space-x-2">
                    {getAccuracyIcon(result.accuracy)}
                    {getStatusBadge(result.status, result.accuracy)}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Predicted */}
                  <div>
                    <h4 className="font-medium text-gray-300 mb-3">Predicted Podium</h4>
                    <div className="space-y-2">
                      {result.predicted.map((driver, i) => (
                        <div key={i} className="flex items-center space-x-3 p-2 rounded bg-blue-900/20">
                          {getPodiumIcon(i)}
                          <span className="text-white">{driver}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Actual */}
                  <div>
                    <h4 className="font-medium text-gray-300 mb-3">Actual Result</h4>
                    <div className="space-y-2">
                      {result.actual.map((driver, i) => (
                        <div
                          key={i}
                          className={`flex items-center space-x-3 p-2 rounded ${
                            result.predicted[i] === driver ? "bg-green-900/20" : "bg-red-900/20"
                          }`}
                        >
                          {getPodiumIcon(i)}
                          <span className="text-white">{driver}</span>
                          {result.predicted[i] === driver && (
                            <Badge className="bg-green-500/20 text-green-400 border-green-500/30 text-xs">✓</Badge>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

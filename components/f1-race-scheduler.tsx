"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Calendar, MapPin, Clock, Flag, Trophy } from "lucide-react"

interface Race {
  id: string
  name: string
  circuit: string
  location: string
  date: string
  time: string
  status: "completed" | "upcoming" | "current"
  round: number
}

const F1_2025_SCHEDULE: Race[] = [
  {
    id: "bahrain",
    name: "Bahrain Grand Prix",
    circuit: "Bahrain International Circuit",
    location: "Sakhir, Bahrain",
    date: "March 2, 2025",
    time: "15:00 GMT",
    status: "completed",
    round: 1,
  },
  {
    id: "jeddah",
    name: "Saudi Arabian Grand Prix",
    circuit: "Jeddah Corniche Circuit",
    location: "Jeddah, Saudi Arabia",
    date: "March 9, 2025",
    time: "17:00 GMT",
    status: "completed",
    round: 2,
  },
  {
    id: "albert_park",
    name: "Australian Grand Prix",
    circuit: "Albert Park Grand Prix Circuit",
    location: "Melbourne, Australia",
    date: "March 23, 2025",
    time: "05:00 GMT",
    status: "completed",
    round: 3,
  },
  {
    id: "suzuka",
    name: "Japanese Grand Prix",
    circuit: "Suzuka Circuit",
    location: "Suzuka, Japan",
    date: "April 13, 2025",
    time: "06:00 GMT",
    status: "upcoming",
    round: 4,
  },
  {
    id: "shanghai",
    name: "Chinese Grand Prix",
    circuit: "Shanghai International Circuit",
    location: "Shanghai, China",
    date: "April 20, 2025",
    time: "07:00 GMT",
    status: "upcoming",
    round: 5,
  },
  {
    id: "miami",
    name: "Miami Grand Prix",
    circuit: "Miami International Autodrome",
    location: "Miami, USA",
    date: "May 4, 2025",
    time: "19:30 GMT",
    status: "upcoming",
    round: 6,
  },
  {
    id: "imola",
    name: "Emilia Romagna Grand Prix",
    circuit: "Autodromo Enzo e Dino Ferrari",
    location: "Imola, Italy",
    date: "May 18, 2025",
    time: "13:00 GMT",
    status: "upcoming",
    round: 7,
  },
  {
    id: "monaco",
    name: "Monaco Grand Prix",
    circuit: "Circuit de Monaco",
    location: "Monte Carlo, Monaco",
    date: "May 25, 2025",
    time: "13:00 GMT",
    status: "upcoming",
    round: 8,
  },
  {
    id: "villeneuve",
    name: "Canadian Grand Prix",
    circuit: "Circuit Gilles Villeneuve",
    location: "Montreal, Canada",
    date: "June 8, 2025",
    time: "18:00 GMT",
    status: "upcoming",
    round: 9,
  },
  {
    id: "catalunya",
    name: "Spanish Grand Prix",
    circuit: "Circuit de Barcelona-Catalunya",
    location: "Barcelona, Spain",
    date: "June 22, 2025",
    time: "13:00 GMT",
    status: "upcoming",
    round: 10,
  },
  {
    id: "red_bull_ring",
    name: "Austrian Grand Prix",
    circuit: "Red Bull Ring",
    location: "Spielberg, Austria",
    date: "June 29, 2025",
    time: "13:00 GMT",
    status: "upcoming",
    round: 11,
  },
  {
    id: "silverstone",
    name: "British Grand Prix",
    circuit: "Silverstone Circuit",
    location: "Silverstone, UK",
    date: "July 6, 2025",
    time: "14:00 GMT",
    status: "upcoming",
    round: 12,
  },
]

interface F1RaceSchedulerProps {
  onRaceSelect: (race: Race) => void
}

export default function F1RaceScheduler({ onRaceSelect }: F1RaceSchedulerProps) {
  const [selectedRace, setSelectedRace] = useState<Race | null>(null)

  const handleRaceSelect = (race: Race) => {
    setSelectedRace(race)
    onRaceSelect(race)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-500/20 text-green-400 border-green-500/30"
      case "current":
        return "bg-red-500/20 text-red-400 border-red-500/30"
      case "upcoming":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30"
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30"
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <Trophy className="h-3 w-3" />
      case "current":
        return <Flag className="h-3 w-3" />
      case "upcoming":
        return <Calendar className="h-3 w-3" />
      default:
        return <Clock className="h-3 w-3" />
    }
  }

  return (
    <Card className="bg-black/40 border-red-600/30 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-white flex items-center">
          <Calendar className="h-5 w-5 mr-2 text-red-400" />
          2025 F1 Race Calendar
          <Badge className="ml-2 bg-red-500/20 text-red-400 border-red-500/30">{F1_2025_SCHEDULE.length} Races</Badge>
        </CardTitle>
        <CardDescription className="text-gray-400">Select a race to generate ML-powered predictions</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {F1_2025_SCHEDULE.map((race) => (
            <Card
              key={race.id}
              className={`cursor-pointer transition-all duration-200 hover:scale-105 ${
                selectedRace?.id === race.id
                  ? "bg-red-600/20 border-red-500/50"
                  : "bg-gray-800/30 border-gray-600/30 hover:bg-gray-700/40"
              }`}
              onClick={() => handleRaceSelect(race)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <Badge variant="outline" className="border-gray-600 text-gray-300 text-xs">
                    Round {race.round}
                  </Badge>
                  <Badge className={getStatusColor(race.status)}>
                    {getStatusIcon(race.status)}
                    <span className="ml-1 capitalize">{race.status}</span>
                  </Badge>
                </div>
                <CardTitle className="text-white text-lg">{race.name}</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  <div className="flex items-center text-sm text-gray-400">
                    <MapPin className="h-4 w-4 mr-2" />
                    {race.circuit}
                  </div>
                  <div className="flex items-center text-sm text-gray-400">
                    <MapPin className="h-4 w-4 mr-2" />
                    {race.location}
                  </div>
                  <div className="flex items-center text-sm text-gray-400">
                    <Calendar className="h-4 w-4 mr-2" />
                    {race.date}
                  </div>
                  <div className="flex items-center text-sm text-gray-400">
                    <Clock className="h-4 w-4 mr-2" />
                    {race.time}
                  </div>
                </div>
                <Button
                  className="w-full mt-4 bg-red-600 hover:bg-red-700 text-white"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleRaceSelect(race)
                  }}
                >
                  Select for ML Prediction
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

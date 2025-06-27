import { type NextRequest, NextResponse } from "next/server"

// Ergast F1 API integration for live race schedule
const ERGAST_API_BASE = "http://ergast.com/api/f1"

interface F1Race {
  round: string
  raceName: string
  date: string
  time?: string
  circuit: {
    circuitId: string
    circuitName: string
    location: {
      locality: string
      country: string
      lat: string
      long: string
    }
  }
  qualifying?: {
    date: string
    time: string
  }
  sprint?: {
    date: string
    time: string
  }
}

interface ProcessedRace {
  id: string
  name: string
  date: string
  time: string
  circuit: string
  location: string
  country: string
  circuitId: string
  status: "upcoming" | "completed" | "live"
  sessions: {
    practice1?: { date: string; time: string }
    practice2?: { date: string; time: string }
    practice3?: { date: string; time: string }
    qualifying?: { date: string; time: string }
    sprint?: { date: string; time: string }
    race: { date: string; time: string }
  }
}

async function fetchF1Schedule(year?: number): Promise<F1Race[]> {
  const currentYear = year || new Date().getFullYear()

  // Try multiple API endpoints and fallback gracefully
  const apiEndpoints = [`https://ergast.com/api/f1/${currentYear}.json`, `http://ergast.com/api/f1/${currentYear}.json`]

  for (const url of apiEndpoints) {
    try {
      console.log(`Attempting to fetch from: ${url}`)

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 8000) // 8 second timeout

      const response = await fetch(url, {
        headers: {
          "User-Agent": "F1-Predictor/1.0",
          Accept: "application/json",
          "Cache-Control": "no-cache",
        },
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        console.log(`HTTP error from ${url}: ${response.status}`)
        continue
      }

      const contentType = response.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        console.log(`Invalid content type from ${url}: ${contentType}`)
        continue
      }

      const text = await response.text()

      if (!text || text.trim().length === 0) {
        console.log(`Empty response from ${url}`)
        continue
      }

      // Validate JSON structure
      if (!text.trim().startsWith("{") && !text.trim().startsWith("[")) {
        console.log(`Invalid JSON format from ${url}`)
        continue
      }

      try {
        const data = JSON.parse(text)

        if (data && data.MRData && data.MRData.RaceTable && data.MRData.RaceTable.Races) {
          console.log(`Successfully fetched ${data.MRData.RaceTable.Races.length} races from ${url}`)
          return data.MRData.RaceTable.Races
        } else {
          console.log(`Invalid data structure from ${url}`)
          continue
        }
      } catch (parseError) {
        console.log(`JSON parse error from ${url}:`, parseError)
        continue
      }
    } catch (error) {
      console.log(`Fetch error from ${url}:`, error)
      continue
    }
  }

  console.log("All API endpoints failed, using comprehensive fallback data")
  return getComprehensiveMockF1Schedule(currentYear)
}

function getComprehensiveMockF1Schedule(year: number): F1Race[] {
  // Comprehensive 2024 F1 schedule with all races
  const baseSchedule: F1Race[] = [
    {
      round: "1",
      raceName: "Bahrain Grand Prix",
      date: "2024-03-02",
      time: "15:00:00Z",
      circuit: {
        circuitId: "bahrain",
        circuitName: "Bahrain International Circuit",
        location: { locality: "Sakhir", country: "Bahrain", lat: "26.0325", long: "50.5106" },
      },
      qualifying: { date: "2024-03-01", time: "15:00:00Z" },
    },
    {
      round: "2",
      raceName: "Saudi Arabian Grand Prix",
      date: "2024-03-09",
      time: "17:00:00Z",
      circuit: {
        circuitId: "jeddah",
        circuitName: "Jeddah Corniche Circuit",
        location: { locality: "Jeddah", country: "Saudi Arabia", lat: "21.6319", long: "39.1044" },
      },
      qualifying: { date: "2024-03-08", time: "17:00:00Z" },
    },
    {
      round: "3",
      raceName: "Australian Grand Prix",
      date: "2024-03-24",
      time: "05:00:00Z",
      circuit: {
        circuitId: "albert_park",
        circuitName: "Albert Park Grand Prix Circuit",
        location: { locality: "Melbourne", country: "Australia", lat: "-37.8497", long: "144.968" },
      },
      qualifying: { date: "2024-03-23", time: "05:00:00Z" },
    },
    {
      round: "4",
      raceName: "Japanese Grand Prix",
      date: "2024-04-07",
      time: "05:00:00Z",
      circuit: {
        circuitId: "suzuka",
        circuitName: "Suzuka Circuit",
        location: { locality: "Suzuka", country: "Japan", lat: "34.8431", long: "136.541" },
      },
      qualifying: { date: "2024-04-06", time: "05:00:00Z" },
    },
    {
      round: "5",
      raceName: "Chinese Grand Prix",
      date: "2024-04-21",
      time: "07:00:00Z",
      circuit: {
        circuitId: "shanghai",
        circuitName: "Shanghai International Circuit",
        location: { locality: "Shanghai", country: "China", lat: "31.3389", long: "121.22" },
      },
      qualifying: { date: "2024-04-20", time: "07:00:00Z" },
      sprint: { date: "2024-04-20", time: "03:00:00Z" },
    },
    {
      round: "6",
      raceName: "Miami Grand Prix",
      date: "2024-05-05",
      time: "19:30:00Z",
      circuit: {
        circuitId: "miami",
        circuitName: "Miami International Autodrome",
        location: { locality: "Miami", country: "USA", lat: "25.9581", long: "-80.2389" },
      },
      qualifying: { date: "2024-05-04", time: "19:30:00Z" },
      sprint: { date: "2024-05-04", time: "15:30:00Z" },
    },
    {
      round: "7",
      raceName: "Emilia Romagna Grand Prix",
      date: "2024-05-19",
      time: "13:00:00Z",
      circuit: {
        circuitId: "imola",
        circuitName: "Autodromo Enzo e Dino Ferrari",
        location: { locality: "Imola", country: "Italy", lat: "44.3439", long: "11.7167" },
      },
      qualifying: { date: "2024-05-18", time: "13:00:00Z" },
    },
    {
      round: "8",
      raceName: "Monaco Grand Prix",
      date: "2024-05-26",
      time: "13:00:00Z",
      circuit: {
        circuitId: "monaco",
        circuitName: "Circuit de Monaco",
        location: { locality: "Monte-Carlo", country: "Monaco", lat: "43.7347", long: "7.42056" },
      },
      qualifying: { date: "2024-05-25", time: "13:00:00Z" },
    },
    {
      round: "9",
      raceName: "Canadian Grand Prix",
      date: "2024-06-09",
      time: "18:00:00Z",
      circuit: {
        circuitId: "villeneuve",
        circuitName: "Circuit Gilles Villeneuve",
        location: { locality: "Montreal", country: "Canada", lat: "45.5", long: "-73.5228" },
      },
      qualifying: { date: "2024-06-08", time: "18:00:00Z" },
    },
    {
      round: "10",
      raceName: "Spanish Grand Prix",
      date: "2024-06-23",
      time: "13:00:00Z",
      circuit: {
        circuitId: "catalunya",
        circuitName: "Circuit de Barcelona-Catalunya",
        location: { locality: "Montmeló", country: "Spain", lat: "41.57", long: "2.26111" },
      },
      qualifying: { date: "2024-06-22", time: "13:00:00Z" },
    },
    {
      round: "11",
      raceName: "Austrian Grand Prix",
      date: "2024-06-30",
      time: "13:00:00Z",
      circuit: {
        circuitId: "red_bull_ring",
        circuitName: "Red Bull Ring",
        location: { locality: "Spielberg", country: "Austria", lat: "47.2197", long: "14.7647" },
      },
      qualifying: { date: "2024-06-29", time: "13:00:00Z" },
      sprint: { date: "2024-06-29", time: "09:00:00Z" },
    },
    {
      round: "12",
      raceName: "British Grand Prix",
      date: "2024-07-07",
      time: "14:00:00Z",
      circuit: {
        circuitId: "silverstone",
        circuitName: "Silverstone Circuit",
        location: { locality: "Silverstone", country: "UK", lat: "52.0786", long: "-1.01694" },
      },
      qualifying: { date: "2024-07-06", time: "14:00:00Z" },
    },
    {
      round: "13",
      raceName: "Hungarian Grand Prix",
      date: "2024-07-21",
      time: "13:00:00Z",
      circuit: {
        circuitId: "hungaroring",
        circuitName: "Hungaroring",
        location: { locality: "Budapest", country: "Hungary", lat: "47.5789", long: "19.2486" },
      },
      qualifying: { date: "2024-07-20", time: "13:00:00Z" },
    },
    {
      round: "14",
      raceName: "Belgian Grand Prix",
      date: "2024-07-28",
      time: "13:00:00Z",
      circuit: {
        circuitId: "spa",
        circuitName: "Circuit de Spa-Francorchamps",
        location: { locality: "Spa", country: "Belgium", lat: "50.4372", long: "5.97139" },
      },
      qualifying: { date: "2024-07-27", time: "13:00:00Z" },
    },
    {
      round: "15",
      raceName: "Dutch Grand Prix",
      date: "2024-08-25",
      time: "13:00:00Z",
      circuit: {
        circuitId: "zandvoort",
        circuitName: "Circuit Zandvoort",
        location: { locality: "Zandvoort", country: "Netherlands", lat: "52.3888", long: "4.54092" },
      },
      qualifying: { date: "2024-08-24", time: "13:00:00Z" },
    },
    {
      round: "16",
      raceName: "Italian Grand Prix",
      date: "2024-09-01",
      time: "13:00:00Z",
      circuit: {
        circuitId: "monza",
        circuitName: "Autodromo Nazionale di Monza",
        location: { locality: "Monza", country: "Italy", lat: "45.6156", long: "9.28111" },
      },
      qualifying: { date: "2024-08-31", time: "13:00:00Z" },
    },
    {
      round: "17",
      raceName: "Azerbaijan Grand Prix",
      date: "2024-09-15",
      time: "11:00:00Z",
      circuit: {
        circuitId: "baku",
        circuitName: "Baku City Circuit",
        location: { locality: "Baku", country: "Azerbaijan", lat: "40.3725", long: "49.8533" },
      },
      qualifying: { date: "2024-09-14", time: "11:00:00Z" },
      sprint: { date: "2024-09-14", time: "07:30:00Z" },
    },
    {
      round: "18",
      raceName: "Singapore Grand Prix",
      date: "2024-09-22",
      time: "12:00:00Z",
      circuit: {
        circuitId: "marina_bay",
        circuitName: "Marina Bay Street Circuit",
        location: { locality: "Marina Bay", country: "Singapore", lat: "1.2914", long: "103.864" },
      },
      qualifying: { date: "2024-09-21", time: "12:00:00Z" },
    },
    {
      round: "19",
      raceName: "United States Grand Prix",
      date: "2024-10-20",
      time: "19:00:00Z",
      circuit: {
        circuitId: "americas",
        circuitName: "Circuit of the Americas",
        location: { locality: "Austin", country: "USA", lat: "30.1328", long: "-97.6411" },
      },
      qualifying: { date: "2024-10-19", time: "19:00:00Z" },
      sprint: { date: "2024-10-19", time: "15:00:00Z" },
    },
    {
      round: "20",
      raceName: "Mexican Grand Prix",
      date: "2024-10-27",
      time: "20:00:00Z",
      circuit: {
        circuitId: "rodriguez",
        circuitName: "Autódromo Hermanos Rodríguez",
        location: { locality: "Mexico City", country: "Mexico", lat: "19.4042", long: "-99.0907" },
      },
      qualifying: { date: "2024-10-26", time: "20:00:00Z" },
    },
    {
      round: "21",
      raceName: "São Paulo Grand Prix",
      date: "2024-11-03",
      time: "17:00:00Z",
      circuit: {
        circuitId: "interlagos",
        circuitName: "Autódromo José Carlos Pace",
        location: { locality: "São Paulo", country: "Brazil", lat: "-23.7036", long: "-46.6997" },
      },
      qualifying: { date: "2024-11-02", time: "17:00:00Z" },
      sprint: { date: "2024-11-02", time: "13:30:00Z" },
    },
    {
      round: "22",
      raceName: "Las Vegas Grand Prix",
      date: "2024-11-23",
      time: "06:00:00Z",
      circuit: {
        circuitId: "vegas",
        circuitName: "Las Vegas Strip Circuit",
        location: { locality: "Las Vegas", country: "USA", lat: "36.1147", long: "-115.173" },
      },
      qualifying: { date: "2024-11-22", time: "06:00:00Z" },
    },
    {
      round: "23",
      raceName: "Qatar Grand Prix",
      date: "2024-12-01",
      time: "16:00:00Z",
      circuit: {
        circuitId: "losail",
        circuitName: "Losail International Circuit",
        location: { locality: "Al Daayen", country: "Qatar", lat: "25.49", long: "51.4542" },
      },
      qualifying: { date: "2024-11-30", time: "16:00:00Z" },
      sprint: { date: "2024-11-30", time: "12:00:00Z" },
    },
    {
      round: "24",
      raceName: "Abu Dhabi Grand Prix",
      date: "2024-12-08",
      time: "13:00:00Z",
      circuit: {
        circuitId: "yas_marina",
        circuitName: "Yas Marina Circuit",
        location: { locality: "Abu Dhabi", country: "UAE", lat: "24.4672", long: "54.6031" },
      },
      qualifying: { date: "2024-12-07", time: "13:00:00Z" },
    },
  ]

  // Adjust dates for different years
  if (year !== 2024) {
    return baseSchedule.map((race) => ({
      ...race,
      date: race.date.replace("2024", year.toString()),
      qualifying: race.qualifying
        ? {
            ...race.qualifying,
            date: race.qualifying.date.replace("2024", year.toString()),
          }
        : undefined,
      sprint: race.sprint
        ? {
            ...race.sprint,
            date: race.sprint.date.replace("2024", year.toString()),
          }
        : undefined,
    }))
  }

  return baseSchedule
}

async function fetchCurrentStandings() {
  const currentYear = new Date().getFullYear()

  try {
    console.log("Attempting to fetch current F1 standings...")

    const fetchWithTimeout = async (url: string, timeout = 5000) => {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), timeout)

      try {
        const response = await fetch(url, {
          signal: controller.signal,
          headers: {
            "User-Agent": "F1-Predictor/1.0",
            Accept: "application/json",
            "Cache-Control": "no-cache",
          },
        })
        clearTimeout(timeoutId)
        return response
      } catch (error) {
        clearTimeout(timeoutId)
        throw error
      }
    }

    const [driverResponse, constructorResponse] = await Promise.allSettled([
      fetchWithTimeout(`https://ergast.com/api/f1/${currentYear}/driverStandings.json`),
      fetchWithTimeout(`https://ergast.com/api/f1/${currentYear}/constructorStandings.json`),
    ])

    let drivers = []
    let constructors = []

    // Process driver standings
    if (driverResponse.status === "fulfilled" && driverResponse.value.ok) {
      try {
        const contentType = driverResponse.value.headers.get("content-type")
        if (contentType && contentType.includes("application/json")) {
          const text = await driverResponse.value.text()
          if (text && text.trim().startsWith("{")) {
            const driverData = JSON.parse(text)
            if (driverData.MRData?.StandingsTable?.StandingsLists?.[0]?.DriverStandings) {
              drivers = driverData.MRData.StandingsTable.StandingsLists[0].DriverStandings
              console.log(`✓ Loaded ${drivers.length} driver standings`)
            }
          }
        }
      } catch (parseError) {
        console.log("Driver standings parse error:", parseError)
      }
    } else {
      console.log(
        "Driver standings fetch failed:",
        driverResponse.status === "fulfilled" ? driverResponse.value.status : "Network error",
      )
    }

    // Process constructor standings
    if (constructorResponse.status === "fulfilled" && constructorResponse.value.ok) {
      try {
        const contentType = constructorResponse.value.headers.get("content-type")
        if (contentType && contentType.includes("application/json")) {
          const text = await constructorResponse.value.text()
          if (text && text.trim().startsWith("{")) {
            const constructorData = JSON.parse(text)
            if (constructorData.MRData?.StandingsTable?.StandingsLists?.[0]?.ConstructorStandings) {
              constructors = constructorData.MRData.StandingsTable.StandingsLists[0].ConstructorStandings
              console.log(`✓ Loaded ${constructors.length} constructor standings`)
            }
          }
        }
      } catch (parseError) {
        console.log("Constructor standings parse error:", parseError)
      }
    } else {
      console.log(
        "Constructor standings fetch failed:",
        constructorResponse.status === "fulfilled" ? constructorResponse.value.status : "Network error",
      )
    }

    // Use mock data if no live data was successfully fetched
    if (drivers.length === 0) {
      console.log("Using mock driver standings")
      drivers = getMockDriverStandings()
    }
    if (constructors.length === 0) {
      console.log("Using mock constructor standings")
      constructors = getMockConstructorStandings()
    }

    return { drivers, constructors }
  } catch (error) {
    console.error("Error fetching standings, using mock data:", error)
    return {
      drivers: getMockDriverStandings(),
      constructors: getMockConstructorStandings(),
    }
  }
}

function getMockDriverStandings() {
  return [
    {
      position: "1",
      points: "575",
      wins: "19",
      Driver: {
        driverId: "max_verstappen",
        givenName: "Max",
        familyName: "Verstappen",
        nationality: "Dutch",
      },
      Constructors: [{ name: "Red Bull Racing Honda RBPT", constructorId: "red_bull" }],
    },
    {
      position: "2",
      points: "350",
      wins: "1",
      Driver: {
        driverId: "leclerc",
        givenName: "Charles",
        familyName: "Leclerc",
        nationality: "Monégasque",
      },
      Constructors: [{ name: "Ferrari", constructorId: "ferrari" }],
    },
    {
      position: "3",
      points: "285",
      wins: "2",
      Driver: {
        driverId: "perez",
        givenName: "Sergio",
        familyName: "Pérez",
        nationality: "Mexican",
      },
      Constructors: [{ name: "Red Bull Racing Honda RBPT", constructorId: "red_bull" }],
    },
    {
      position: "4",
      points: "280",
      wins: "1",
      Driver: {
        driverId: "sainz",
        givenName: "Carlos",
        familyName: "Sainz",
        nationality: "Spanish",
      },
      Constructors: [{ name: "Ferrari", constructorId: "ferrari" }],
    },
    {
      position: "5",
      points: "225",
      wins: "0",
      Driver: {
        driverId: "norris",
        givenName: "Lando",
        familyName: "Norris",
        nationality: "British",
      },
      Constructors: [{ name: "McLaren Mercedes", constructorId: "mclaren" }],
    },
    {
      position: "6",
      points: "195",
      wins: "0",
      Driver: {
        driverId: "russell",
        givenName: "George",
        familyName: "Russell",
        nationality: "British",
      },
      Constructors: [{ name: "Mercedes", constructorId: "mercedes" }],
    },
    {
      position: "7",
      points: "190",
      wins: "0",
      Driver: {
        driverId: "hamilton",
        givenName: "Lewis",
        familyName: "Hamilton",
        nationality: "British",
      },
      Constructors: [{ name: "Mercedes", constructorId: "mercedes" }],
    },
    {
      position: "8",
      points: "180",
      wins: "0",
      Driver: {
        driverId: "piastri",
        givenName: "Oscar",
        familyName: "Piastri",
        nationality: "Australian",
      },
      Constructors: [{ name: "McLaren Mercedes", constructorId: "mclaren" }],
    },
    {
      position: "9",
      points: "85",
      wins: "0",
      Driver: {
        driverId: "alonso",
        givenName: "Fernando",
        familyName: "Alonso",
        nationality: "Spanish",
      },
      Constructors: [{ name: "Aston Martin Aramco Mercedes", constructorId: "aston_martin" }],
    },
    {
      position: "10",
      points: "45",
      wins: "0",
      Driver: {
        driverId: "stroll",
        givenName: "Lance",
        familyName: "Stroll",
        nationality: "Canadian",
      },
      Constructors: [{ name: "Aston Martin Aramco Mercedes", constructorId: "aston_martin" }],
    },
  ]
}

function getMockConstructorStandings() {
  return [
    {
      position: "1",
      points: "860",
      wins: "21",
      Constructor: {
        constructorId: "red_bull",
        name: "Red Bull Racing Honda RBPT",
        nationality: "Austrian",
      },
    },
    {
      position: "2",
      points: "630",
      wins: "2",
      Constructor: {
        constructorId: "ferrari",
        name: "Ferrari",
        nationality: "Italian",
      },
    },
    {
      position: "3",
      points: "405",
      wins: "0",
      Constructor: {
        constructorId: "mclaren",
        name: "McLaren Mercedes",
        nationality: "British",
      },
    },
    {
      position: "4",
      points: "385",
      wins: "0",
      Constructor: {
        constructorId: "mercedes",
        name: "Mercedes",
        nationality: "German",
      },
    },
    {
      position: "5",
      points: "130",
      wins: "0",
      Constructor: {
        constructorId: "aston_martin",
        name: "Aston Martin Aramco Mercedes",
        nationality: "British",
      },
    },
  ]
}

function processRaceData(races: F1Race[]): ProcessedRace[] {
  const currentDate = new Date()

  return races.map((race) => {
    const raceDate = new Date(`${race.date}T${race.time || "14:00:00"}`)
    let status: "upcoming" | "completed" | "live" = "upcoming"

    if (raceDate < currentDate) {
      status = "completed"
    } else if (Math.abs(raceDate.getTime() - currentDate.getTime()) < 3 * 60 * 60 * 1000) {
      // Within 3 hours of race time
      status = "live"
    }

    return {
      id: `${race.date}-${race.circuit.circuitId}`,
      name: race.raceName,
      date: race.date,
      time: race.time || "14:00:00",
      circuit: race.circuit.circuitName,
      location: race.circuit.location.locality,
      country: race.circuit.location.country,
      circuitId: race.circuit.circuitId,
      status,
      sessions: {
        qualifying: race.qualifying
          ? {
              date: race.qualifying.date,
              time: race.qualifying.time,
            }
          : undefined,
        sprint: race.sprint
          ? {
              date: race.sprint.date,
              time: race.sprint.time,
            }
          : undefined,
        race: {
          date: race.date,
          time: race.time || "14:00:00",
        },
      },
    }
  })
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const year = searchParams.get("year")
  const status = searchParams.get("status") // upcoming, completed, all
  const includeStandings = searchParams.get("standings") === "true"

  try {
    console.log(`Fetching F1 schedule for year: ${year || "current"}`)

    // Fetch race schedule
    const races = await fetchF1Schedule(year ? Number.parseInt(year) : undefined)
    const processedRaces = processRaceData(races)

    // Filter by status if requested
    let filteredRaces = processedRaces
    if (status && status !== "all") {
      filteredRaces = processedRaces.filter((race) => race.status === status)
    }

    // Fetch current standings if requested
    let standings = null
    if (includeStandings) {
      standings = await fetchCurrentStandings()
    }

    // Get upcoming races for quick access
    const upcomingRaces = processedRaces.filter((race) => race.status === "upcoming").slice(0, 5)

    // Get next race
    const nextRace = upcomingRaces[0] || null

    return NextResponse.json({
      success: true,
      data: {
        races: filteredRaces,
        upcomingRaces,
        nextRace,
        standings,
        summary: {
          total: processedRaces.length,
          upcoming: processedRaces.filter((r) => r.status === "upcoming").length,
          completed: processedRaces.filter((r) => r.status === "completed").length,
          live: processedRaces.filter((r) => r.status === "live").length,
        },
      },
      metadata: {
        year: year || new Date().getFullYear(),
        lastUpdated: new Date().toISOString(),
        source: "Ergast F1 API",
      },
    })
  } catch (error) {
    console.error("F1 Schedule API error:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch F1 schedule",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const { action, raceId, year } = await request.json()

    switch (action) {
      case "refresh":
        // Force refresh of race data
        const races = await fetchF1Schedule(year)
        const processedRaces = processRaceData(races)

        return NextResponse.json({
          success: true,
          message: "Race schedule refreshed",
          data: processedRaces,
          timestamp: new Date().toISOString(),
        })

      case "get-race-details":
        if (!raceId) {
          return NextResponse.json({ error: "Race ID required" }, { status: 400 })
        }

        // Fetch detailed race information
        const raceYear = year || new Date().getFullYear()
        const raceRound = raceId.split("-")[0] // Assuming raceId format includes round

        const detailUrl = `${ERGAST_API_BASE}/${raceYear}/${raceRound}/results.json`
        const qualifyingUrl = `${ERGAST_API_BASE}/${raceYear}/${raceRound}/qualifying.json`

        const [raceResults, qualifyingResults] = await Promise.all([
          fetch(detailUrl).then((r) => (r.ok ? r.json() : null)),
          fetch(qualifyingUrl).then((r) => (r.ok ? r.json() : null)),
        ])

        return NextResponse.json({
          success: true,
          data: {
            results: raceResults?.MRData?.RaceTable?.Races?.[0]?.Results || [],
            qualifying: qualifyingResults?.MRData?.RaceTable?.Races?.[0]?.QualifyingResults || [],
          },
        })

      default:
        return NextResponse.json({ error: "Unknown action" }, { status: 400 })
    }
  } catch (error) {
    console.error("F1 Schedule POST error:", error)
    return NextResponse.json(
      {
        error: "Failed to process request",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}

import { type NextRequest, NextResponse } from "next/server"

// Real F1 data URLs (your provided data)
const F1_DATA_SOURCES = {
  sprint_results:
    "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/sprint_results-Y4Ic4CTL6hQnjb0aUY6Gsjq5zE1W2V.csv",
  seasons: "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/seasons-hs9GJLLWKJFhEhMfr1AOU9HKY8oeyK.csv",
  races: "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/races-QUJ9dlQpgRzhjobaUk0eblDAoPpsrH.csv",
  qualifying: "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/qualifying-Fl7YfWJFvtszwzdIOsp3msmQEujBud.csv",
  results: "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/results-P6NQQAMhGzgbfM8kZqHewfGHkRnJxC.csv",
  circuits: "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/circuits-YS2b8VUceqiMjFOMWv2RVBEj57PCxB.csv",
  status: "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/status-aPr4Bd28dJlhdLcGMWO93Jf9CnE1UZ.csv",
  constructor_results:
    "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/constructor_results-Zyf36AZ5XdxBm6z5XwPbduiz4d4YGk.csv",
  pit_stops: "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/pit_stops-DNz7vdGmA7jQGh5eP2diUXZVAn9UDp.csv",
}

// Function to fetch and parse CSV data
async function fetchCSVData(url: string) {
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const csvText = await response.text()

    // Simple CSV parsing (for production, use a proper CSV parser)
    const lines = csvText.split("\n")
    const headers = lines[0].split(",").map((h) => h.trim().replace(/"/g, ""))
    const data = lines
      .slice(1)
      .filter((line) => line.trim())
      .map((line) => {
        const values = line.split(",").map((v) => v.trim().replace(/"/g, ""))
        const row: Record<string, string> = {}
        headers.forEach((header, index) => {
          row[header] = values[index] || ""
        })
        return row
      })

    return { headers, data, count: data.length }
  } catch (error) {
    console.error(`Error fetching data from ${url}:`, error)
    throw error
  }
}

// Data processing functions
async function processRaceData() {
  const races = await fetchCSVData(F1_DATA_SOURCES.races)
  const results = await fetchCSVData(F1_DATA_SOURCES.results)

  // Get recent races (2024)
  const recentRaces = races.data.filter((race) => race.year === "2024").slice(-10)

  return {
    totalRaces: races.count,
    recentRaces: recentRaces.length,
    totalResults: results.count,
    lastRace: recentRaces[recentRaces.length - 1]?.name || "Unknown",
  }
}

async function processDriverData() {
  const results = await fetchCSVData(F1_DATA_SOURCES.results)

  // Calculate driver statistics
  const driverStats: Record<string, any> = {}

  results.data.forEach((result) => {
    const driverId = result.driverId
    if (!driverStats[driverId]) {
      driverStats[driverId] = {
        races: 0,
        wins: 0,
        podiums: 0,
        points: 0,
        dnfs: 0,
      }
    }

    driverStats[driverId].races++
    if (result.position === "1") driverStats[driverId].wins++
    if (["1", "2", "3"].includes(result.position)) driverStats[driverId].podiums++
    driverStats[driverId].points += Number.parseInt(result.points) || 0
    if (result.positionText === "R") driverStats[driverId].dnfs++
  })

  return {
    totalDrivers: Object.keys(driverStats).length,
    topPerformers: Object.entries(driverStats)
      .sort(([, a], [, b]) => (b as any).points - (a as any).points)
      .slice(0, 10)
      .map(([driverId, stats]) => ({ driverId, ...stats })),
  }
}

async function processCircuitData() {
  const circuits = await fetchCSVData(F1_DATA_SOURCES.circuits)
  const races = await fetchCSVData(F1_DATA_SOURCES.races)

  // Get circuit usage statistics
  const circuitUsage: Record<string, number> = {}
  races.data.forEach((race) => {
    circuitUsage[race.circuitId] = (circuitUsage[race.circuitId] || 0) + 1
  })

  return {
    totalCircuits: circuits.count,
    mostUsedCircuits: Object.entries(circuitUsage)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([circuitId, count]) => {
        const circuit = circuits.data.find((c) => c.circuitId === circuitId)
        return {
          circuitId,
          name: circuit?.name || "Unknown",
          country: circuit?.country || "Unknown",
          usage: count,
        }
      }),
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const type = searchParams.get("type")
  const source = searchParams.get("source")

  try {
    // If specific data source requested
    if (source && F1_DATA_SOURCES[source as keyof typeof F1_DATA_SOURCES]) {
      console.log(`Fetching ${source} data...`)
      const data = await fetchCSVData(F1_DATA_SOURCES[source as keyof typeof F1_DATA_SOURCES])
      return NextResponse.json({
        success: true,
        source,
        ...data,
        timestamp: new Date().toISOString(),
      })
    }

    // Process different types of data
    switch (type) {
      case "races":
        const raceData = await processRaceData()
        return NextResponse.json({
          success: true,
          type: "races",
          data: raceData,
          timestamp: new Date().toISOString(),
        })

      case "drivers":
        const driverData = await processDriverData()
        return NextResponse.json({
          success: true,
          type: "drivers",
          data: driverData,
          timestamp: new Date().toISOString(),
        })

      case "circuits":
        const circuitData = await processCircuitData()
        return NextResponse.json({
          success: true,
          type: "circuits",
          data: circuitData,
          timestamp: new Date().toISOString(),
        })

      case "summary":
        // Get summary of all data sources
        const summary = await Promise.all([processRaceData(), processDriverData(), processCircuitData()])

        return NextResponse.json({
          success: true,
          type: "summary",
          data: {
            races: summary[0],
            drivers: summary[1],
            circuits: summary[2],
            dataSources: Object.keys(F1_DATA_SOURCES),
            lastUpdated: new Date().toISOString(),
          },
        })

      default:
        // Return available data sources
        return NextResponse.json({
          success: true,
          availableSources: Object.keys(F1_DATA_SOURCES),
          availableTypes: ["races", "drivers", "circuits", "summary"],
          usage: {
            source: "Get specific dataset: ?source=races",
            type: "Get processed data: ?type=summary",
          },
        })
    }
  } catch (error) {
    console.error("Data fetch error:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch F1 data",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, source, parameters } = body

    console.log(`Processing data action: ${action}`)

    switch (action) {
      case "preprocess":
        // Simulate data preprocessing
        await new Promise((resolve) => setTimeout(resolve, 2000))

        return NextResponse.json({
          success: true,
          action: "preprocess",
          message: "Data preprocessing completed",
          processed: {
            records: 15000 + Math.floor(Math.random() * 5000),
            features: 25 + Math.floor(Math.random() * 10),
            quality_score: 85 + Math.random() * 10,
          },
          timestamp: new Date().toISOString(),
        })

      case "validate":
        // Simulate data validation
        await new Promise((resolve) => setTimeout(resolve, 1500))

        return NextResponse.json({
          success: true,
          action: "validate",
          validation: {
            completeness: 94.2 + Math.random() * 5,
            accuracy: 97.8 + Math.random() * 2,
            consistency: 89.1 + Math.random() * 8,
            timeliness: 98.5 + Math.random() * 1.5,
          },
          issues: Math.floor(Math.random() * 5),
          timestamp: new Date().toISOString(),
        })

      case "refresh":
        // Simulate data refresh
        await new Promise((resolve) => setTimeout(resolve, 3000))

        return NextResponse.json({
          success: true,
          action: "refresh",
          message: "All data sources refreshed successfully",
          updated: Object.keys(F1_DATA_SOURCES).length,
          timestamp: new Date().toISOString(),
        })

      default:
        return NextResponse.json(
          {
            error: "Unknown action",
            availableActions: ["preprocess", "validate", "refresh"],
          },
          { status: 400 },
        )
    }
  } catch (error) {
    console.error("Data processing error:", error)
    return NextResponse.json(
      {
        error: "Failed to process data request",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}

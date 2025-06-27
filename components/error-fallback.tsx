"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertTriangle, RefreshCw } from "lucide-react"

interface ErrorFallbackProps {
  error?: string
  onRetry?: () => void
  title?: string
  description?: string
}

export default function ErrorFallback({
  error,
  onRetry,
  title = "Unable to Load Data",
  description = "We're having trouble connecting to the F1 data sources. Using offline data instead.",
}: ErrorFallbackProps) {
  return (
    <Card className="bg-black/40 border-yellow-600/30 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-white flex items-center">
          <AlertTriangle className="h-5 w-5 mr-2 text-yellow-400" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-gray-400">{description}</p>

        {error && (
          <div className="p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
            <p className="text-red-400 text-sm font-mono">{error}</p>
          </div>
        )}

        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
          <span className="text-yellow-400 text-sm">Using offline F1 schedule data</span>
        </div>

        {onRetry && (
          <Button
            onClick={onRetry}
            variant="outline"
            className="border-yellow-600 text-yellow-400 hover:bg-yellow-600/20 bg-transparent"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Try Again
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

"use client"

import { useState, useEffect } from "react"
import F1FancyLoading from "@/components/f1-fancy-loading"
import F12025EnhancedPredictor from "@/components/f1-2025-enhanced-predictor"

export default function Home() {
  const [isLoading, setIsLoading] = useState(true)
  const [modelAccuracy, setModelAccuracy] = useState(87.3)

  useEffect(() => {
    // Simulate initial app loading
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 5000) // 5 second loading experience

    return () => clearTimeout(timer)
  }, [])

  const handleLoadingComplete = () => {
    setIsLoading(false)
  }

  if (isLoading) {
    return <F1FancyLoading onComplete={handleLoadingComplete} duration={5000} />
  }

  return (
    <div className="min-h-screen">
      <F12025EnhancedPredictor />
    </div>
  )
}

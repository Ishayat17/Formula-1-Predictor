#!/bin/bash

echo "🚀 Starting Formula 1 Predictor deployment to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if user is logged in
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please login to Vercel..."
    vercel login
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the project
echo "🔨 Building project..."
npm run build

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment completed!"
echo "🌐 Your app should be live at the URL provided above"
echo "📊 Check the Vercel dashboard for deployment status and logs"

# Commit and push all changes
echo "📤 Pushing changes to GitHub..."
git add .
git commit -m "Update: Replace local code with latest for Vercel deployment"
git push origin main
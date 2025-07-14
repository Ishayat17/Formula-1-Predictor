# Formula 1 Predictor - Vercel Deployment Guide

## 🚀 Deploying to Vercel

### Prerequisites
- Vercel CLI installed: `npm i -g vercel`
- Git repository with your project
- Vercel account

### Step 1: Prepare Your Project

1. **Ensure all files are committed to Git:**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment with updated ML models"
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

### Step 2: Deploy to Vercel

#### Option A: Using Vercel CLI (Recommended)

1. **Login to Vercel:**
   ```bash
   vercel login
   ```

2. **Deploy the project:**
   ```bash
   vercel --prod
   ```

3. **Follow the prompts:**
   - Link to existing project (if you have a previous deployment)
   - Or create a new project
   - Confirm deployment settings

#### Option B: Using Vercel Dashboard

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your Git repository
4. Configure build settings:
   - Framework Preset: Next.js
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

### Step 3: Configure Environment Variables (if needed)

In your Vercel dashboard:
1. Go to Project Settings → Environment Variables
2. Add any environment variables your app needs

### Step 4: Verify Deployment

1. **Check the deployment URL** provided by Vercel
2. **Test the ML prediction endpoint:**
   ```bash
   curl -X POST https://your-app.vercel.app/api/ml-predict \
     -H "Content-Type: application/json" \
     -d '{"race": "Monaco Grand Prix", "year": 2024}'
   ```

### Step 5: Update Custom Domain (if applicable)

If you have a custom domain:
1. Go to Project Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed

## 🔧 Troubleshooting

### Common Issues:

1. **Python dependencies not found:**
   - Ensure `requirements.txt` is in the root directory
   - Check that `runtime.txt` specifies Python 3.9

2. **ML models not loading:**
   - Models will be trained during the first API call
   - Check Vercel function logs for errors

3. **Build failures:**
   - Check the build logs in Vercel dashboard
   - Ensure all dependencies are properly listed

### Monitoring:

1. **Vercel Dashboard:**
   - Monitor function execution times
   - Check for cold start issues
   - Review error logs

2. **Function Logs:**
   - Go to Functions tab in Vercel dashboard
   - Check individual function logs for debugging

## 📊 Performance Optimization

### For ML Models:
- Models are cached after first training
- Consider using Vercel's Edge Functions for faster response times
- Monitor memory usage of large model files

### For Next.js App:
- Images are optimized automatically
- Static assets are served from CDN
- API routes are serverless functions

## 🔄 Updating the Deployment

To update your deployment:

1. **Make your changes locally**
2. **Commit to Git:**
   ```bash
   git add .
   git commit -m "Update Formula 1 Predictor"
   git push
   ```

3. **Redeploy:**
   ```bash
   vercel --prod
   ```

Or if using automatic deployments, just push to your main branch.

## 📈 Scaling Considerations

- **Serverless Functions:** Automatically scale based on demand
- **Cold Starts:** First request may be slower due to model loading
- **Memory Limits:** Be aware of Vercel's function memory limits
- **Timeout Limits:** Functions have a 10-second timeout by default

## 🎯 Success Metrics

After deployment, verify:
- ✅ App loads without errors
- ✅ ML prediction API responds correctly
- ✅ All UI components work as expected
- ✅ Performance is acceptable
- ✅ No console errors in browser

---

**Need help?** Check Vercel's documentation or contact support through the dashboard. 
/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    serverComponentsExternalPackages: ['python-shell'],
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push('python-shell');
    }
    return config;
  },
}

export default nextConfig

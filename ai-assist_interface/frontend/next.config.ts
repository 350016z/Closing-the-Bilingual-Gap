import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  allowedDevOrigins: [
    'http://203.145.216.166:50073/',
  ],

  async rewrites() {
    return [
      {
        // 前端呼叫 /annotation/xxx 時，Next.js 幫你轉發到後端 8000
        source: '/annotation/:path*',
        destination: 'http://127.0.0.1:8000/annotation/:path*',
      },
      {
        // 同樣把 annotation_files 的靜態檔也 proxy 過去
        source: '/annotation_files/:path*',
        destination: 'http://127.0.0.1:8000/annotation_files/:path*',
      },
    ];
  },
};

export default nextConfig;

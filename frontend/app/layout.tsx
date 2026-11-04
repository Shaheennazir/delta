import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "delta — video intel",
  description: "Drop a clip. Ask anything. Verified answers with grounded evidence.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#060608] text-[#f4f1ea] antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "delta — Coaching Analytics",
  description: "Autonomous football drill and coaching analytics review.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-white antialiased">{children}</body>
    </html>
  );
}

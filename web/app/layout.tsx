import type { Metadata } from "next";
import "./style.css";

export const metadata: Metadata = { title: "Netflix Streaming Cockpit", description: "Adaptive HLS playback and explainable live recommendations" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

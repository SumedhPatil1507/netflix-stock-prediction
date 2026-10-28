import * as React from "react";
import { cn } from "@/lib/utils";

export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(function Card(
  { className, ...props }, ref,
) {
  return <div ref={ref} className={cn("rounded-xl border border-white/10 bg-card text-foreground shadow-sm", className)} {...props} />;
});

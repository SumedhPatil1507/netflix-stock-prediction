import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonStyles = cva("inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 disabled:pointer-events-none disabled:opacity-50", {
  variants: {
    variant: { default: "bg-red-600 text-white hover:bg-red-500", outline: "border border-white/15 bg-transparent text-white hover:bg-white/5", ghost: "text-white hover:bg-white/10" },
    size: { default: "h-10 px-4 py-2", sm: "h-8 px-3 text-xs", icon: "h-9 w-9" },
  },
  defaultVariants: { variant: "default", size: "default" },
});

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonStyles> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant, size, ...props }, ref,
) {
  return <button ref={ref} className={cn(buttonStyles({ variant, size }), className)} {...props} />;
});

FROM node:20-alpine AS dependencies
WORKDIR /app
COPY web/package.json ./package.json
COPY web/package-lock.json ./package-lock.json
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY web ./
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ARG NEXT_PUBLIC_LEGACY_UI_URL=http://localhost:8501
ARG NEXT_PUBLIC_DEMO_HLS_URL=
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL \
    NEXT_PUBLIC_LEGACY_UI_URL=$NEXT_PUBLIC_LEGACY_UI_URL \
    NEXT_PUBLIC_DEMO_HLS_URL=$NEXT_PUBLIC_DEMO_HLS_URL
RUN npm run build

FROM nginx:1.27-alpine AS runner
COPY docker/nginx-web.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/out /usr/share/nginx/html
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]

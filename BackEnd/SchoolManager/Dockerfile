# ── Build stage ──────────────────────────────────────
FROM gradle:8.6-jdk17 AS build
WORKDIR /app
# Cache dependencies layer separately
COPY ../../../school-management-gradle/school-management-gradle/build.gradle settings.gradle ./
RUN gradle dependencies --no-daemon || true
COPY ../../../school-management-gradle/school-management-gradle/src ./src
RUN gradle bootJar --no-daemon -x test

# ── Runtime stage ─────────────────────────────────────
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=build /app/build/libs/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]

library(readxl)

pop <- read_excel("Skola/Machine learning project 3.xlsx")

library(corrplot)

cordata <- pop[, c("Population", "Birth rate", "Death rate", "Life Expectancy", "Immigration", "Emigration")]

# Compute the correlation matrix
cormatrix <- cor(cordata)
print(cormatrix)

# Create a heatmap of the correlation matrix
corrplot(cormatrix, method = "color", type = "lower", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black")

# Plot the population 1950-2022
plot(pop$Year, pop$Population, main="Population growth 1950-2022", xlab="Year", ylab="Population")

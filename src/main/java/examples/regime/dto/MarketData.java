package examples.regime.dto;

import java.time.LocalDateTime;

public class MarketData {
    private LocalDateTime timestamp;

    // Core & Macro Levels
    private double usdjpyPrice;
    private double usdjpyAtr;
    private double usdjpyEr; // Added to match CSV
    private double usBondLevel;
    private double jpBondLevel;
    private double wtiOil;

    // Trajectories / Deltas
    private double atrDelta;
    private double erDelta;
    private double yieldSpreadDelta;

    // Cross-Pair Context
    private double eurusdAtr;
    private double audjpyAtr; // Renamed from VOL to ATR to match CSV
    private double eurjpyAtr; // Added to match CSV

    // Dynamically Injected Fields (Updated via ModelTrainer / Strategy Node)
    private double usdCpiShock = 0.0;
    private double usdRatesShock = 0.0;
    private double strategyPerf = 0.0;



    private int regimeLabel = -1; //(Assigned by RegimeLabeler)

    // Getters and Setters
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

    public double getUsdjpyPrice() { return usdjpyPrice; }
    public void setUsdjpyPrice(double usdjpyPrice) { this.usdjpyPrice = usdjpyPrice; }

    public double getUsdjpyAtr() { return usdjpyAtr; }
    public void setUsdjpyAtr(double usdjpyAtr) { this.usdjpyAtr = usdjpyAtr; }

    public double getUsdjpyEr() { return usdjpyEr; }
    public void setUsdjpyEr(double usdjpyEr) { this.usdjpyEr = usdjpyEr; }

    public double getUsBondLevel() { return usBondLevel; }
    public void setUsBondLevel(double usBondLevel) { this.usBondLevel = usBondLevel; }

    public double getJpBondLevel() { return jpBondLevel; }
    public void setJpBondLevel(double jpBondLevel) { this.jpBondLevel = jpBondLevel; }

    public double getWtiOil() { return wtiOil; }
    public void setWtiOil(double wtiOil) { this.wtiOil = wtiOil; }

    public double getAtrDelta() { return atrDelta; }
    public void setAtrDelta(double atrDelta) { this.atrDelta = atrDelta; }

    public double getErDelta() { return erDelta; }
    public void setErDelta(double erDelta) { this.erDelta = erDelta; }

    public double getYieldSpreadDelta() { return yieldSpreadDelta; }
    public void setYieldSpreadDelta(double yieldSpreadDelta) { this.yieldSpreadDelta = yieldSpreadDelta; }

    public double getEurusdAtr() { return eurusdAtr; }
    public void setEurusdAtr(double eurusdAtr) { this.eurusdAtr = eurusdAtr; }

    public double getAudjpyAtr() { return audjpyAtr; }
    public void setAudjpyAtr(double audjpyAtr) { this.audjpyAtr = audjpyAtr; }

    public double getEurjpyAtr() { return eurjpyAtr; }
    public void setEurjpyAtr(double eurjpyAtr) { this.eurjpyAtr = eurjpyAtr; }

    public double getUsdCpiShock() { return usdCpiShock; }
    public void setUsdCpiShock(double usdCpiShock) { this.usdCpiShock = usdCpiShock; }

    public double getUsdRatesShock() { return usdRatesShock; }
    public void setUsdRatesShock(double usdRatesShock) { this.usdRatesShock = usdRatesShock; }

    public double getStrategyPerf() { return strategyPerf; }
    public void setStrategyPerf(double strategyPerf) { this.strategyPerf = strategyPerf; }

    public int getRegimeLabel() {
        return regimeLabel;
    }

    public void setRegimeLabel(int regimeLabel) {
        this.regimeLabel = regimeLabel;
    }
}

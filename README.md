\# Rainfall-to-Streamflow Modeling of Hurricane Florence Flooding in the Trent River Watershed, North Carolina



\## Project Overview



This project investigates the hydrologic response of the Trent River watershed near Trenton, North Carolina, to rainfall associated with Hurricane Florence in September 2018.



The project develops a reproducible Python and HEC-HMS workflow to:



\- Acquire and quality-check observed USGS streamflow data.

\- Process tropical cyclone rainfall inputs.

\- Simulate watershed rainfall-runoff response.

\- Compare observed and simulated hydrographs.

\- Generate synthetic storm scenarios.

\- Evaluate changes in peak discharge and runoff volume under varying rainfall and antecedent soil-moisture conditions.



\## Study Site



| Item | Selection |

|---|---|

| Watershed | Trent River watershed upstream of Trenton, North Carolina |

| USGS Gauge | 02092500 — Trent River near Trenton, NC |

| Drainage Area | 168 square miles |

| Basin | Neuse River Basin, HUC 03020204 |

| Storm Event | Hurricane Florence, September 2018 |

| Analysis Window | September 10–22, 2018 |

| Peak Flood Date | September 16, 2018 |



\## Research Question



How did Hurricane Florence rainfall affect streamflow in the Trent River watershed near Trenton, North Carolina, and how might changes in rainfall intensity and antecedent hydrologic conditions alter peak discharge and runoff volume?



\## Project Workflow



1\. Download and quality-check USGS observed streamflow data.

2\. Retrieve and process tropical cyclone rainfall data.

3\. Delineate or obtain the watershed boundary and basin characteristics.

4\. Construct a rainfall-runoff model in HEC-HMS.

5\. Compare simulated and observed hydrographs.

6\. Generate synthetic rainfall scenarios.

7\. Produce a scenario-based streamflow response dataset.



\## Data Sources



\- U.S. Geological Survey Water Data for the Nation: observed discharge at gauge 02092500.

\- NOAA/National Weather Service: Hurricane Florence event documentation.

\- NASA GPM IMERG or NOAA precipitation data: rainfall inputs.

\- USGS or HEC-HMS geospatial tools: watershed characteristics.



\## Repository Structure



```text

data/

&#x20; raw/

&#x20;   streamflow/

&#x20;   rainfall/

&#x20;   watershed/

&#x20; processed/

figures/

hec\_hms/

notebooks/

references/

results/

src/


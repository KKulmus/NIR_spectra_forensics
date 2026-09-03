## Motivation
Anhand der vorliegenden Daten von Kranenburg et al., der NIR-Spektren verschiedener Substanzen von 5 verschiedenen Spektrometern aufgenommen, enthält, soll in diesem Projekt untersucht werden, ob mittels NIR-Spektren die Substanzklasse korrekt erkannt werden kann. Dies ist interessant für ein erstes On-Site Drug-Checking, bei dem die Frage geklärt werden kann, ob eine Probe bereits durch das Spektrum eindeutig identifiziert ist, oder ob eine chemische (oder physikalische) Analyse im Labor ergänzend gemacht werden muss. Tragbare NIR-Spektrometer, im Datensatz sind es 4, die in unterschiedlichen Wellenlängenbereichen arbeiten, haben den Vorteil, dass sie einfach einzusetzen sind und relativ günstig zu erwerben sind. Die Herausforderung ist, eine Datenbank mit Vergleichsspektren zu nutzen, die Datenqualität zu sichern, Algorithmen zur Vorverarbeitung zu erstellen, ein geeignetes Machine-Learning Modell zu entwickeln und es über die verschiedenen Instrumente zu evaluieren. 
## Data
Research Article: Kranenburg, Ruben F. et al. "The importance of wavelength selection in on-scene identification of drugs of abuse with portable near-infrared spectroscopy" DOI: 10.1016/j.forc.2022.100437

Data Article: Kranenburg, Ruben F. et al. "Dataset of near-infrared spectral data of illicit-drugs and forensic casework samples analyzed by five portable spectrometers operating in different wavelength ranges" DOI: 10.1016/j.dib.2022.108660


Datensatz: DOI: 10.21942/uva.21252300 (Lizenz CC-BY)


Die Daten sind aus lizenzrechtlichen Gründen nicht Teil dieses Repositories und werden über den oben genannten DOI bezogen.

### Instrumente (NIR Spektrometer)

|Instrument|Wellenlängenbereich (nm)| Rohdatenausgabe (Datenpunkte)| Auflösung (FWHM)|
| :---- | :-----| :----- | :----- |
ASD LabSpec 4 | 350-2500 |2151 | 10|
NeoSpectra | 1300-2600|160|16|
NIRone 2.0 | 1550-1950|201|15-21|
MicroNIR|950-1650|125|12.5|
SCiO|740-1070|331|n.a.|

### Sample-Sets

|Symbol|Erläuterung|Anzahl|
| :---- | :----- | :----- |
|C| Common drugs|17|
|D|Designer drugs|38|
|N|Cocaine-negative samples, adulterants (powdered, white)|40|
|PAM|Police Amsterdam powdered casework samples|104|
|K|Calibration set of binary cocaine mixtures|88|
|T|Tablets|72|
|P|Crushed, powdered tablets|71|

**Gesamt: 430 Proben**, verteilt auf 5 Instrumente bzw. 24 Spektraldateien





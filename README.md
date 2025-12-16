# Capstone


### OBS: Det tar lång tid att skapa imagen, kanske 10 minuter pga att den laddar ner tunga  paket.


1. Jag la till pypdf i requirements för att kunna splitta pdf dokumentet för att kunna testa köra lite saker. 

2. Jag la till easyocr för att DoclingParser behöver det för document som inte bara är text och det kommer inte med datapizza.

3. Det verkar som att datapizza har någon standardinställning på Qdrant som vi inte har, så vi måste sätta vår collection till att använda modellnamnet text-embedding-3-small.

4. Istället för att använda memory som collection, så kan man skapa en container i docker där allting sparas, vilket innebär att man inte måste göra om alla 90 sidor till vektorer varje gång koden körs (vilket tar typ en timme), så man gör det bara en gång och sparar det i containern/ens data för den är mountad. Den nya containern använder qdrant imagen, som är en färdig image. 

5. Kör ingest 1 gång för att få vektorer, med resvaneundersokning_test, alltså den som innehåller 2 sidor, denna collection är döpt till my_document. Sedan kör jag ingest 1 gång till med hela rapporten på 90 sidor och denna döper jag till resvaneundersokning_document, så nu finns det 2 stycket dokument i Qdrant- databasen som ligger i en separat container. 


# Konspekt
	1. Opis problemu
		Praca polegać będzie na przeprowadzeniu detekcji anomalii na podstawie lokalizacji opublikowanych tweetów w czasie rzeczywistym.
    Po wykryciu ponadprzeciętnej liczby tweetów z danego miejsca przy użyciu NLP z biblioteki Transformers zostanie przeprowadzana próba 
    podsumowania czego te twitty dotyczą. Praca ma na celu stworzenie zautomatyzowanego systemu, który w czasie rzeczywistym będzie w stanie 
    klasyfikować istotne, bądź "gorące" wydarzenia. Dodatkowo można stworzyć własny model NLP na historycznych danych twitterowych i porównać 
    jak się sprawdza w zestawieniu z uniwersalnie wyuczonymi Transformersami. 
    
	2. Uzasadnienie biznesowe
		Stworzony system mógłby służyć użytkownikowi za swoisty agregat ważnych wydarzeń z okolicy, co może mieć zastosowanie w przypadku np. nagłych wypadków, 
    zagrożeń (chociażby na przykładzie obecnej wojny w Ukrainie - posty świadków przelotu rosyjskich pocisków świadczące o zbliżającym się bombardowaniu) 
    ale także "głośnych" wydarzeń jak koncerty, marsze, przemówienia polityków. 
    
	3. Hipoteza badawcza
  
	4. Wykorzystane technologie:
		PySpark / Flink i Kafka 
    
	5. Algorytmy/ modele:
		- Algorytm detekcji anomalii
		- Gotowe wytrepnowane modele NLP z biblioteki Transfomers 
	
# Układ pracy:
	1. Wstęp - opis zastosowania docelowego systemu, wskazanie motywacji oraz uzasadnienia biznesowego podjęcia tego tematu. 
    Wymienienie hipotez, metod badawczych oraz opis układu pracy
  
	2. Przegląd literatury dotyczącej detekcji anomalii, przedstawienie wykorzystywanego algorytmu
  
	3. Przegląd literatury dotyczącej NLP, opis HuggingFace, Transformersów, konkretnego wykorzystywanego modelu
  
	4. Porównanie przetwarzania wsadowego i strumieniowego. Opis strumieni danych. Opis wykorzystanej architektury przepływu danych.
    Opis wykorzystanych narzędzi. Apache Spark vs Apache Flink. 
  
	5. Przedstawienie wyników działania algorytmu detekcji anomalii i grupowania postów. Przedstawienie wyników modelu NLP.  
    (Dodatkowo można porównać wyniki Transofmersów z relatywnie mniej złożonym modelem wyuczonym na danych wyłącznie twitterowych). 
  
	6. Wnioski - jak sprawdził się algorytm detekcjo anomalii oraz Transfomersy, czy metody dają satysfakcjonujące wyniki? 
    Czy serwis twitter jest odpowiednim źródłem dla takiego systemu. Czy wybrany Spark/Flink sprawdził się jako framework do przetwarzania strumieniowego?

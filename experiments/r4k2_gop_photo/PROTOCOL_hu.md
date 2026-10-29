# R4K-2: a kódolt frame-típus hatása éles etalonfotókon (kontrollált GOP-próba)

## Protokoll (rögzítve 2026-10-28, mielőtt bármely eredmény megvolt)

**Előzmény.** Az R4K-1 két videón megfigyelte, hogy a modellek a snitten belüli I-frame-et élesebbnek pontozzák P/B
szomszédjainál. Ez megfigyelés volt: a valódi élességet ott nem ismertük. A `round4_terv.md` 4b próbája pontos
jellel ismétli: 0 px-es etalonfotókból készült klipeken minden frame éles, így minden blur-döntés téves riasztás.
A ChatGPT-beszélgetés (2026-10-28) két kiegészítését átvesszük: a csak I-frame-es kódolás páros kontrollként, és a
minőség rögzítése frame-enként (azonos CRF nem jelent azonos minőséget).

**Kérdés.** Ugyanaz az éles forrásframe kap-e több blur-döntést vagy magasabb pontszámot pusztán attól, hogy P- vagy
B-frame-ként kódoltuk, nem I-frame-ként?

**Modellek** (publikált fájlok, publikált betöltők):
- round3 Base (`round3_base_p99_ep029.pt`) és round3 Small (`round3_small_p99_ep029.pt`), Update 1 betöltő;
- v9 Base (`v9_base_selfmap_v2_ep019.pt`), Update 2 betöltő, beépített M (belső cellák).

**Fotók.** A fotólétra 300 etalonfotójának sorrendjében (validációs és teszt rész, [két album, a nevük visszatartva], sosem
tanított) az első 100 fekvő, 1620 px széles fotó. Kivágás: 1440 × 1080.

**Klipek** (fotónként kettő, 48 frame, 30 fps):
- álló: ugyanaz a kivágás 48-szor;
- pásztázás: a kivágás frame-enként 1 px-lel jobbra lép (egész pixel: nincs újramintavételezés, minden frame 0 px).

**Kódolás** (a tanítás láncával: bt709, tv tartomány, yuv420p, libx264 high, preset medium, 1 szál):
- három szerkezet:
  - `I`: csak I-frame (keyint 1);
  - `IP`: keyint 15, B-frame nélkül;
  - `IPB`: keyint 15, 3 B-frame.
- Közös: scenecut 0, zárt GOP, így az I-frame-ek a 0., 15., 30. és 45. frame-en vannak, az IP és az IPB klipben is.
- négy ráta: CRF 23 (a fotólétra és a tanítás jellemző értéke), CRF 28, CRF 33, és 2 Mbit/s átlagos bitráta
  (maxrate 2 Mbit/s, puffer 4 Mbit; 1440 × 1080-on ez nagyjából a 2,6 Mbit/s-os 1080p filmnek felel meg).
- A frame-típust ffprobe olvassa a kódolt folyamból (`pict_type`); a dekódolt frame-ek a megjelenítési sorrendben
  jönnek, a számuk ellenőrzött (48).
- Minőség: frame-enként a PSNR a forráshoz (RGB, 8 bit) és a kódolt méret (`pkt_size`).

Összesen 100 fotó × 2 klip × 3 szerkezet × 4 ráta × 48 frame = 115 200 frame, mindhárom modellel pontozva (GPU,
bfloat16 autocast, fingerprint float32).

### Mértékek

1. **Téves riasztás frame-típus szerint.** A blurnak mondott arány szerkezet, ráta, mozgás és frame-típus szerint:
   round3 modellek a saját r95 (és r98) küszöbükön, a v9 M > 0,007882 (r95) és M > 0 mellett.
2. **Páros különbség az I-kontrollal (elsődleges).** Ugyanaz a forrásframe (fotó, klip, ráta, sorszám) az `IP` vagy
   `IPB` kódolásban és a csak `I` kódolásban: a pontszám különbsége (round3: logit, v9: M) és a blurnak mondott arány
   különbsége, az `IP` / `IPB` frame típusa szerint bontva. 95%-os intervallum bootstrappal a fotókra (2000
   újramintavétel, seed 20261028).
3. **Az R4K-1 megismétlése.** Az `IPB` klipek klipen belüli I-frame-jei (15., 30., 45.) a két szomszédjukkal: hány
   esetben a hármas legélesebbje (legalacsonyabb pontszám) az I-frame. Null: ugyanezek a pozíciók a csak `I`
   kódolásban (ott mindhárom I-frame). Döntetlen külön számolva.
4. **Azonos minőségnél.** A páros különbség PSNR-sávonként (1 dB): ha a P/B frame-ek hatása csak a rosszabb
   minőségből jön, azonos PSNR-nél eltűnik.

### Értelmezés (előre)

- **A kodekhatás igazolt**, ha legalább egy rátán a P/B frame-ek blurnak mondott aránya (vagy pontszáma) nagyobb,
  mint ugyanazoké a forrásframe-eké csak I-kódolással, és a páros intervallum nem tartalmazza a 0-t. A nagyságot
  rátánként jelentjük.
- Ha igazolt, a `round4_terv.md` 4b következményei: a fotók többframe-es kódolással is menjenek át a tanításban; a
  filmes kiértékelés frame-típus szerint bontandó; a II. fázis ilyen kódolásnál I-frame-et választana.
- Ha egyik rátán sem mutatható ki, a filmes jelenség (R4K-1) nem a frame-típus önmagában: a valódi filmes tartalom
  (mozgás, zaj, fény) vagy a kódolók eltérése marad magyarázatnak.
- Az álló klip a legkedvezőbb eset a P/B frame-eknek (szinte mindent a referenciából másolnak), a pásztázás a
  mozgásbecslést is próbára teszi. Valódi mozgáselmosódás egyikben sincs.

### Nincs benne

- Valódi filmes anyag, más kódolók (x265, hardveres), más felbontás, tört pixeles mozgás: ha az eredmény indokolja,
  külön próba.

## Napló

./r420-stats | collector/collector | tee >(stdbuf -oL grep 'FH Table entry' >> recorded_data_2.fh.log) >(stdbuf -oL grep 'Measurement entry' >> recorded_data_2.log) >/dev/null;

echo "[none]" >> recorded_data_2.log
echo "[klawiatura]" >> recorded_data_2.log
echo "[zdobywcy]" >> recorded_data_2.log
echo "[krzeslo]" >> recorded_data_2.log
echo "[obudowa]" >> recorded_data_2.log
echo "[szuflady]" >> recorded_data_2.log
echo "[salewa]" >> recorded_data_2.log
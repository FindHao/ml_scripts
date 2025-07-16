for ((X_BLOCK = 32; X_BLOCK <= 4096; X_BLOCK *= 2)); do
  for ((Y_BLOCK = 32; Y_BLOCK <= 4096; Y_BLOCK *= 2)); do
    echo "==============Running with X_BLOCK=$X_BLOCK and Y_BLOCK=$Y_BLOCK"
    # python run_opt2_for_triton.py --XBLOCK $X_BLOCK --YBLOCK $Y_BLOCK
    env XBLOCK=$X_BLOCK YBLOCK=$Y_BLOCK python run_opt2.py
  done
done

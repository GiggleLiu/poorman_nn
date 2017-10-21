sed -i "s/Parameters/Args/1" *.py
sed -i "s/Return:/Returns:/1" *.py
sed -i "s/:\([a-zA-Z0-9_]\+\): \(.\+\),/\1 (\2):/1" *.py

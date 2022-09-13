sources=('3821')

# python3 main.py --task lipschitz --numimage 3 --targettype 1 --modelfile models/mnist_7layer_relu_1024 --layerbndalg crown-adaptive \
# --jacbndalg fastlip --norm i --lipsteps 80 --liplogstart -3 --liplogend 0 --dataset mnist --gpu 2 --eps 0.1

for i in "${!sources[@]}"
do
    :
    python3 main.py --task lipschitz --numimage 3 --targettype ${sources[$i]} --modelfile models/mnist_7layer_relu_1024 --layerbndalg crown-adaptive \
    --jacbndalg fastlip --norm i --lipsteps 80 --liplogstart -3 --liplogend 0 --dataset vggwhite --gpu 2 --eps 0.1

    python3 main.py --task lipschitz --numimage 3 --targettype ${sources[$i]} --modelfile models/mnist_7layer_relu_1024 --layerbndalg fastlin-interval \
    --jacbndalg fastlip --norm i --lipsteps 80 --liplogstart -3 --liplogend 0 --dataset vggwhite --gpu 2 --eps 0.1

    python3 main.py --task lipschitz --numimage 3 --targettype ${sources[$i]} --modelfile models/mnist_7layer_relu_1024 --layerbndalg crown-adaptive \
    --jacbndalg recurjac --norm i --lipsteps 80 --liplogstart -3 --liplogend 0 --dataset vggwhite --gpu 2 --eps 0.1

    python3 main.py --task lipschitz --numimage 3 --targettype ${sources[$i]} --modelfile models/mnist_7layer_relu_1024 --layerbndalg fastlin-interval \
    --jacbndalg recurjac --norm i --lipsteps 80 --liplogstart -3 --liplogend 0 --dataset vggwhite --gpu 2 --eps 0.1
done

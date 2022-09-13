sources=("200" "201" "202" "203" "204" "205" "206" "207" "208" "209" "210" "211" "212" "213" "214" "215" "216" "217" "218" "219" "220" "221" "222" "223" "224" "225" "226" "227" "228" "229" "230" "231" "232" "233" "234" "235" "236" "237" "238" "239" "240" "241" "242" "243" "244" "245" "246" "247" "248" "249" "250" "251" "252" "253" "254" "255" "256" "257" "258" "259" "260" "261" "262" "263" "264" "265" "266" "267" "268" "269" "270" "271" "272" "273" "274" "275" "276" "277" "278" "279" "280" "281" "282" "283" "284" "285" "286" "287" "288" "289" "290" "291" "292" "293" "294" "295" "296" "297" "298" "299" "300" "301" "302" "303" "304" "305" "306" "307" "308" "309" "310" "311" "312" "313" "314" "315" "316" "317" "318" "319" "320" "321" "322" "323" "324" "325" "326" "327" "328" "329" "330" "331" "332" "333" "334" "335" "336" "337" "338" "339" "340" "341" "342" "343" "344" "345" "346" "347" "348" "349" "350" "351" "352" "353" "354" "355" "356" "357" "358" "359" "360" "361" "362" "363" "364" "365" "366" "367" "368" "369" "370" "371" "372" "373" "374" "375" "376" "377" "378" "379" "380" "381" "382" "383" "384" "385" "386" "387" "388" "389" "390" "391" "392" "393" "394" "395" "396" "397" "398" "399")
samples=('0' '1' '2')


for i in "${!sources[@]}"
do
    :
    for j in "${!samples[@]}"
    do
        :
        python3 main.py --task lipschitz --numimage 1 --targettype ${sources[$i]} --modelfile models/vgg-bal-sex-vgg16.hdf5 --layerbndalg fastlin-interval \
        --jacbndalg recurjac --norm i --lipsteps 20 --liplogstart -3 --liplogend 0 --dataset vgg-bal-sex --gpu 7 --eps 0.1 --exactsample ${samples[$j]}
    done
done

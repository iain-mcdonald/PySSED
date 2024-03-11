for I in {1..14196}
	do
		FILENAME=`head -6 ${I}.asc | awk '$0~"teff" {teff=$4} $0~"logg" {logg=$4} $0~"meta" {meta=$4} $0~"alpha" {alpha=$4} FNR==6 {printf "t%05ig%+0.2fm%+0.2fa%+0.2f.dat\n",teff,logg,meta,alpha}'`
		echo "$I" "$FILENAME"
		awk '$1+0>0 {print $1,$2}' ${I}.asc > ${FILENAME}
		rm ${I}.asc
	done

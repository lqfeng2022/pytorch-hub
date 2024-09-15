import chapterFive from '../../data/chapterFive'
import Definition from '../Definition'
import BaseGrid from '../BaseGrid'

const Backpropagation = () => {
  const [ defin, implem, calcul ] = chapterFive[4].sections

  return (
    <>
      <Definition title={defin.name} definition={defin}/>
      <BaseGrid section={implem}/>
      <BaseGrid section={calcul}/>
    </>
  )
}

export default Backpropagation
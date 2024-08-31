import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import chapterZero from '../../data/chapterZero'

const NNSection = () => {
  const [ whats, model, neurons ] = chapterZero[3].sections
  
  return (
    <>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={model}/>
      <BaseGrid section={neurons}/>
    </>
  )
}

export default NNSection
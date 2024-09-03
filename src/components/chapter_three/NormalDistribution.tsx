import Definition from '../Definition'
import chapterThree from '../../data/chapterThree'
import BaseGrid from '../BaseGrid'

const NormalDistribution = () => {
  const [ whats, pdf, cdf ]= chapterThree[1].sections

  return (
    <div>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={pdf}/>
      <BaseGrid section={cdf}/>
    </div>
  )
}

export default NormalDistribution
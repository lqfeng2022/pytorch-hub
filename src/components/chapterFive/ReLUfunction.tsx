import chapterFive from '../../data/chapterFive'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'

const ReLUfunction = () => {
  const [ defin, formula, features ]= chapterFive[3].sections

  return (
    <div>
      <Definition title={defin.name} definition={defin}/>
      <LeftGrid section={formula}/>
      <BaseGrid section={features}/>
    </div>
  )
}

export default ReLUfunction
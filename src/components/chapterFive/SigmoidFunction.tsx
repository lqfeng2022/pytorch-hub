import chapterFive from '../../data/chapterFive'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'

const SigmoidFunction = () => {
  const [ defin, formula, features ] = chapterFive[2].sections

  return (
    <>
      <Definition title={defin.name} definition={defin}/>
      <LeftGrid section={formula}/>
      <BaseGrid section={features}/>
    </>
  )
}

export default SigmoidFunction
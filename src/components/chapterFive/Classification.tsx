import chapterFive from '../../data/chapterFive'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const BinaryClassification = () => {
  const [ 
    classificDefin, classific, binaryDefin, binaryLinear, binaryNonlinear 
  ] = chapterFive[0].sections

  return (
    <div>
      <Definition title={classificDefin.name} definition={classific}/>
      <BaseGrid section={classific}/>
      <Definition title={binaryDefin.name} definition={binaryDefin}/>
      <LeftGrid section={binaryLinear}/>
      <RightGrid section={binaryNonlinear}/>
    </div>
  )
}

export default BinaryClassification
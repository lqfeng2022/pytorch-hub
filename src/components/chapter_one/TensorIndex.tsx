import chapterOne from '../../data/chapterOne'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const TensorIndex = () => {
  const [ define, basic, slice ] = chapterOne[5].sections
  const [
    indexbasicCode, indexsliceCode 
  ] = pythonCode.slice(18, 20).map(obj => obj.code)

  return (
    <>
      <Definition title={define.name} definition={define}/>
      <LeftGrid section={basic}/>
      <CodeDisplay codes={indexbasicCode}/>
      <RightGrid section={slice}/>
      <CodeDisplay codes={indexsliceCode}/>
    </>
  )
}

export default TensorIndex
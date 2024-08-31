import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'
import Definition from '../Definition'

const TensorIndex = () => {
  const [ define, basic, slice ] = chapterOne[5].sections
  const [
    indexbasicCode, indexsliceCode 
  ] = pythonCode.slice(13, 15).map(obj => obj.code)

  return (
    <>
      <Definition title={define.name} definition={define}/>
      <BaseGrid section={basic}/>
      <CodeDisplay codes={indexbasicCode}/>
      <BaseGrid section={slice}/>
      <CodeDisplay codes={indexsliceCode}/>
    </>
  )
}

export default TensorIndex
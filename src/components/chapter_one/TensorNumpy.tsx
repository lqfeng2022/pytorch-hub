import chapterOne from '../../data/chapterOne'
import pythonCode from '../../data/codeTensors'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'

const TensorNumpy = () => {
  const [tensor_numpy] = chapterOne[6].sections
  const tensornumpyCode = pythonCode[14].code

  return (
    <>
      <BaseGrid section={tensor_numpy}/>
      <CodeDisplay codes={tensornumpyCode}/>
    </>
  )
}

export default TensorNumpy
import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'

const TensorCreate = () => {
  const [ create, random, zerosOnes, range, likes ] =  chapterOne[1].sections
  const [
    randomCode, zerosCode, rangeCode, likesCode
  ] = pythonCode.slice(1, 5).map(obj => obj.code);

  return (
    <>
      <BaseGrid section={create}/>
      <BaseGrid section={random}/>
      <CodeDisplay codes={randomCode}/>
      <BaseGrid section={zerosOnes}/>
      <CodeDisplay codes={zerosCode}/>
      <BaseGrid section={range}/>
      <CodeDisplay codes={rangeCode}/>
      <BaseGrid section={likes}/>
      <CodeDisplay codes={likesCode}/>
    </>
  )
}

export default TensorCreate
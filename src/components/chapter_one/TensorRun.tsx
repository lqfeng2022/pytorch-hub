import chapterOne from '../../data/chapterOne'
import pythonCode from '../../data/pythonCode'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'

const TensorRun = () => {
  const [ 
    ongpu, rungpu, gpu, gpuf, cuda, cudaf, getgpu
  ] = chapterOne[7].sections
  const tensorrunCode = pythonCode[22].code
  
  return (
    <>
      <Definition title={ongpu.name} definition={ongpu}/>
      <BaseGrid section={rungpu}/>
      <CodeDisplay codes={tensorrunCode}/>
      <Definition title={gpu.name} definition={gpu}/>
      <BaseGrid section={gpuf}/>
      <LeftGrid section={cuda}/>
      <BaseGrid section={cudaf}/>
      <BaseGrid section={getgpu}/>
    </>
  )
}

export default TensorRun
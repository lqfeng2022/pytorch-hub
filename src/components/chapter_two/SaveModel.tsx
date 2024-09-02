import Definition from '../Definition'
import chapterTwo from '../../data/chapterTwo'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'
import codeLineModel from '../../data/codeLineModel'
import CodeDisplay from '../CodeDisplay'

const SaveModel = () => {
  const [ define, save, load ] = chapterTwo[4].sections
  const [ savecode, loadcode ] = codeLineModel.slice(10, 12).map(obj => obj.code)

  return (
    <div>
      <Definition title={define.name} definition={define}/>
      <LeftGrid section={save}/>
      <CodeDisplay codes={savecode}/>
      <RightGrid section={load}/>
      <CodeDisplay codes={loadcode}/>
    </div>
  )
}

export default SaveModel
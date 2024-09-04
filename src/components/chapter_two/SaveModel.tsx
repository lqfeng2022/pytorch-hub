import Definition from '../Definition'
import chapterTwo from '../../data/chapterTwo'
import codeLineModel from '../../data/codeLineModel'
import CodeDisplay from '../CodeDisplay'
import BaseGrid from '../BaseGrid'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const SaveModel = () => {
  const [ define, save, load ] = chapterTwo[4].sections
  const [ savecode, loadcode ] = codeLineModel.slice(9, 11).map(obj => obj.code)

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
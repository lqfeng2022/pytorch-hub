import { Box } from '@chakra-ui/react'
import chapterFour from '../../data/chapterFour'
import codeBinaryModel from '../../data/codeBinaryModel'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const TheFirstModel = () => {
  const [ one, one_architec, one_build, one_visual, one_loss ] = chapterFour[4].sections
  const [ one_build_code, one_train_code, one_loss_code
  ] = codeBinaryModel.slice(7, 10).map(obj => obj.code)

  return (
    <Box pt={5}>
      <BaseGrid section={one}/>
      <LeftGrid section={one_architec}/>
      <RightGrid section={one_build}/>
      <CodeDisplay codes={one_build_code}/>
      <LeftGrid section={one_visual}/>
      <CodeDisplay codes={one_train_code}/>
      <RightGrid section={one_loss}/>
      <CodeDisplay codes={one_loss_code}/>
    </Box>
  )
}

export default TheFirstModel
import { Box, Text } from '@chakra-ui/react';
import BaseGrid from '../BaseGrid';
import Definition from '../Definition';
import chapterOne from '../../data/chapterOne';

const PTSection = () => {
  const [ definition, features, trends, companies ] = chapterOne[5].sections

  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl'>6. PyTorch</Text>
        <Definition definition={definition}/>
      </Box>
      <BaseGrid section={features}/>
      <BaseGrid section={trends}/>
      <BaseGrid section={companies}/>
    </>
  )
}

export default PTSection